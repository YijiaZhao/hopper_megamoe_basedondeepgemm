"""Benchmark the current SM90 MXFP4 split MegaMoE kernel."""

import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_mxfp4 import quantize_to_mxfp4, quantize_to_qoq_int4
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist


def _prepare_weights(args: argparse.Namespace, num_local_experts: int, rank_idx: int):
    torch.manual_seed(args.weight_seed + rank_idx * 1000003)
    l1_bf = torch.randn(
        (num_local_experts, 2 * args.intermediate_hidden, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * args.weight_scale
    int4_qoq = os.environ.get("DG_W4A8_INT", "0") != "0"
    l1_packed, l1_scale = (
        quantize_to_qoq_int4(l1_bf) if int4_qoq else
        quantize_to_mxfp4(l1_bf, group_size=32))
    del l1_bf

    l2_bf = torch.randn(
        (num_local_experts, args.hidden, args.intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * args.weight_scale
    l2_packed, l2_scale = (
        quantize_to_qoq_int4(l2_bf) if int4_qoq else
        quantize_to_mxfp4(l2_bf, group_size=32))
    del l2_bf

    transform = (deep_gemm.transform_qoq_int4_weights_for_mega_moe_sm90
                 if int4_qoq else deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90)
    transformed = transform(
        (l1_packed, l1_scale), (l2_packed, l2_scale), block_n=args.block_n)
    del l1_packed, l1_scale, l2_packed, l2_scale
    torch.cuda.empty_cache()
    return transformed


def _benchmark_case(
    args: argparse.Namespace,
    m_tokens: int,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    attn_tp_group: dist.ProcessGroup,
    buffer,
    transformed_l1,
    transformed_l2,
    router_weight,
):
    attn_dp_rank = rank_idx // args.attn_tp_size
    attn_tp_rank = rank_idx % args.attn_tp_size
    # Router weights are TP-replicated, but every physical rank owns a distinct
    # DP token shard and is an independent MegaMoE source.
    torch.manual_seed(args.input_seed + m_tokens * 1009 + rank_idx * 1000003)
    x_bf = torch.randn(
        (m_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    scores = (x_bf.float() @ router_weight.float().t()) if args.fuse_router else torch.randn(
        (m_tokens, args.num_experts), dtype=torch.float32, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, args.num_topk, dim=-1, largest=True, sorted=False)
    if args.fuse_router and args.router_renormalize:
        topk_weights = torch.softmax(topk_weights, dim=-1)
    if not args.fuse_router and os.environ.get('DG_BALANCED_ROUTING', '0') != '0':
        # Every source token contributes one TopK assignment to each EP rank.
        slots = torch.arange(args.num_topk, device='cuda', dtype=torch.int64)
        target_rank = slots % num_ranks
        local_experts = args.num_experts // num_ranks
        token_global = rank_idx * m_tokens + torch.arange(
            m_tokens, device='cuda', dtype=torch.int64)
        expert_offset = (token_global[:, None] * args.num_topk + slots[None, :] * 7) % local_experts
        topk_idx = target_rank[None, :] * local_experts + expert_offset
        topk_weights = torch.full(
            (m_tokens, args.num_topk), 1.0 / args.num_topk,
            device='cuda', dtype=torch.float32)
    if os.environ.get("DG_W4A8_INT", "0") != "0":
        x_float = x_bf.float()
        x_scale = (x_float.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
        x_int8 = (x_float / x_scale).round().clamp(-127, 127).to(torch.int8)
        # Symmetric buffer is FP8-typed for the shared MXFP4 infrastructure;
        # QoQ kernels consume these raw bytes as signed int8.
        x_fp8 = x_int8.view(torch.float8_e4m3fn)
        x_sf = x_scale.repeat(1, args.hidden // 128).contiguous()
    else:
        x_fp8, x_sf = per_token_cast_to_fp8(
            x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)

    num_local_experts = args.num_experts // num_ranks
    phase_profile_enabled = os.environ.get("DG_SM90_MOE_PHASE_PROFILE", "0") != "0"
    phase_profile_ints = 64 if phase_profile_enabled else 0
    stats = torch.zeros(
        num_local_experts + phase_profile_ints,
        dtype=torch.int32,
        device="cuda",
    )
    use_tp_collective = args.tp_combine_allgather or args.tp_combine_broadcast
    y_rows = (m_tokens * args.attn_tp_size
              if args.bf16_e2e and use_tp_collective else m_tokens)
    y = torch.empty((y_rows, args.hidden), dtype=torch.bfloat16, device="cuda")
    tp_combine_output = (
        torch.empty((m_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
        if args.bf16_e2e and use_tp_collective else
        (torch.empty((m_tokens // args.attn_tp_size, args.hidden),
                     dtype=torch.bfloat16, device="cuda")
         if args.tp_combine_allgather else None))
    router_logits_dummy = torch.empty(
        (m_tokens, args.num_experts), dtype=torch.bfloat16, device="cuda")
    router_logits_e2e = torch.empty(
        (m_tokens, args.num_experts), dtype=torch.bfloat16, device="cuda")
    if args.global_scales:
        l1_global_scales = torch.linspace(
            0.73, 1.37, num_local_experts, dtype=torch.float32, device="cuda")
        l2_global_scales = torch.linspace(
            1.31, 0.67, num_local_experts, dtype=torch.float32, device="cuda")
    else:
        l1_global_scales = None
        l2_global_scales = None

    def run():
        if args.bf16_e2e and os.environ.get('DG_BALANCED_ROUTING', '0') != '0':
            # Benchmark-only balanced-routing gate: keep the real fused
            # Router+Quant+TopK cost, then replace routing slots with the
            # deterministic balanced pattern before MegaMoE dispatch.
            if os.environ.get("DG_W4A8_INT", "0") != "0":
                deep_gemm._C.qoq_fused_router_quant_topk(
                    x_bf, router_weight, buffer.x[:m_tokens],
                    buffer.x_sf[:m_tokens], buffer.topk_idx[:m_tokens],
                    buffer.topk_weights[:m_tokens])
            else:
                deep_gemm._C.mxfp4_router_quant_topk(
                    x_bf, router_weight, buffer.x[:m_tokens],
                    buffer.x_sf[:m_tokens], buffer.topk_idx[:m_tokens],
                    buffer.topk_weights[:m_tokens])
            buffer.topk_idx[:m_tokens].copy_(topk_idx)
            buffer.topk_weights[:m_tokens].copy_(topk_weights)
            local_y = tp_combine_output if tp_combine_output is not None else y
            deep_gemm.mxfp4_mega_moe(
                local_y, transformed_l1, transformed_l2, buffer,
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=l1_global_scales,
                l2_global_scales=l2_global_scales,
                recipe=(128, args.block_n, 128), activation="swiglu",
                activation_clamp=args.activation_clamp,
                fast_math=bool(args.fast_math), attn_tp_size=args.attn_tp_size,
                attn_tp_group=attn_tp_group, broadcast_output=False)
            if tp_combine_output is not None:
                dist.all_gather_into_tensor(y, local_y, group=attn_tp_group)
            return y
        if args.bf16_e2e:
            deep_gemm.mxfp4_mega_moe_from_bf16(
                y,
                x_bf,
                router_weight,
                transformed_l1,
                transformed_l2,
                buffer,
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=l1_global_scales,
                l2_global_scales=l2_global_scales,
                recipe=(128, args.block_n, 128),
                activation="swiglu",
                activation_clamp=args.activation_clamp,
                fast_math=bool(args.fast_math),
                attn_tp_size=args.attn_tp_size,
                attn_tp_group=attn_tp_group,
                broadcast_output=args.tp_combine_broadcast,
                router_renormalize=args.router_renormalize,
                tp_combine_output=tp_combine_output,
                router_logits_buffer=router_logits_e2e,
            )
            return y
        runtime_topk_weights = topk_weights
        runtime_topk_idx = topk_idx
        runtime_router_logits = None
        if args.torch_router_fused_topk:
            runtime_router_logits = (
                x_bf @ router_weight.t()
                if (args.router_repeat_tp or attn_tp_rank == 0)
                else router_logits_dummy)
        if args.torch_router and (args.router_repeat_tp or attn_tp_rank == 0):
            router_logits = x_bf @ router_weight.t()
            runtime_topk_weights, runtime_topk_idx = torch.topk(
                router_logits, args.num_topk, dim=-1, largest=True, sorted=False)
            if args.router_renormalize:
                runtime_topk_weights = torch.softmax(runtime_topk_weights.float(), dim=-1)
        buffer.x[:m_tokens].copy_(x_fp8)
        buffer.x_sf[:m_tokens].copy_(x_sf)
        if not args.fuse_router and not args.torch_router_fused_topk:
            buffer.topk_idx[:m_tokens].copy_(runtime_topk_idx)
            buffer.topk_weights[:m_tokens].copy_(runtime_topk_weights)
        deep_gemm.mxfp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            buffer,
            cumulative_local_expert_recv_stats=stats,
            l1_global_scales=l1_global_scales,
            l2_global_scales=l2_global_scales,
            recipe=(128, args.block_n, 128),
            activation="swiglu",
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            attn_tp_size=args.attn_tp_size,
            attn_tp_group=attn_tp_group,
            router_input=x_bf if args.fuse_router else None,
            router_weight=router_weight if args.fuse_router else None,
            router_logits=runtime_router_logits,
            router_renormalize=args.router_renormalize,
            tp_combine_output=tp_combine_output,
        )
        return y

    if args.ncu_profile_only:
        run()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        return

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    if phase_profile_enabled:
        stats.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    if os.environ.get("DG_WALL_BENCH", "0") == "1":
        # Wall-clock: N back-to-back calls under CUDA events. Captures
        # inter-kernel gaps (PDL benefit) that kineto kernel sums cannot.
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start_ev.record()
        for _ in range(args.num_tests):
            run()
        end_ev.record()
        torch.cuda.synchronize()
        wall_us = start_ev.elapsed_time(end_ev) * 1000.0 / args.num_tests
        t = torch.tensor(wall_us, dtype=torch.float64, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        if rank_idx == 0:
            print(f"WALL M={m_tokens} mean_us={t.item() / num_ranks:.3f}", flush=True)
        return

    # with_multiple_kernels sums all matching L1/L2 CUDA time per run().
    elapsed = bench_kineto(
        run,
        "sm90_mxfp4_mega_moe",
        barrier=lambda: dist.barrier(group=group),
        num_tests=args.num_tests,
        suppress_kineto_output=True,
        with_multiple_kernels=True,
    )
    rank_time = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
    rank_min = rank_time.clone()
    rank_max = rank_time.clone()
    rank_sum = rank_time.clone()
    dist.all_reduce(rank_min, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(rank_sum, op=dist.ReduceOp.SUM, group=group)

    if rank_idx == 0:
        print(
            f"RESULT M={m_tokens} block_n={args.block_n} "
            f"rank_min_us={rank_min.item() * 1e6:.3f} "
            f"rank_mean_us={rank_sum.item() / num_ranks * 1e6:.3f} "
            f"rank_max_us={rank_max.item() * 1e6:.3f}",
            flush=True,
        )
        if phase_profile_enabled:
            names = [
                "dispatch_total",
                "dispatch_pull",
                "math_loop",
                "combine_barrier",
                "combine_reduce",
                "gemm_core",
                "l1_epilogue",
                "l2_epilogue",
            ]
            profile = stats[num_local_experts:].view(torch.int64).cpu().tolist()
            num_metrics = len(names)
            for i, name in enumerate(names):
                total = profile[i]
                max_v = profile[num_metrics + i]
                count = profile[2 * num_metrics + i]
                avg = float(total) / count if count else 0.0
                print(
                    f"PHASE {name:16s} avg_cycles={avg:.0f} "
                    f"max_cycles={max_v} count={count}",
                    flush=True,
                )


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if num_ranks % args.attn_tp_size != 0:
        raise ValueError("num_processes must be divisible by attn_tp_size")
    attn_tp_group = None
    for first_rank in range(0, num_ranks, args.attn_tp_size):
        ranks = list(range(first_rank, first_rank + args.attn_tp_size))
        candidate = dist.new_group(ranks)
        if rank_idx in ranks:
            attn_tp_group = candidate
    assert attn_tp_group is not None
    deep_gemm.set_pdl(os.environ.get("DG_PDL", "0") == "1")
    buffer = None
    try:
        if get_arch_major() != 9:
            if rank_idx == 0:
                print(f"[SKIP] requires SM90, got SM{get_arch_major()}0")
            return
        if args.num_experts % num_ranks != 0:
            raise ValueError("num_experts must be divisible by num_processes")

        num_max_tokens_per_rank = max(args.batches)
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            args.num_experts,
            num_max_tokens_per_rank,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
            mma_type="fp8xmxfp4",
            activation="swiglu",
        )
        transformed_l1, transformed_l2 = _prepare_weights(
            args, args.num_experts // num_ranks, rank_idx)
        torch.manual_seed(args.router_seed)
        router_weight = torch.randn(
            (args.num_experts, args.hidden), dtype=torch.bfloat16, device="cuda") * args.weight_scale
        dist.barrier(group=group)

        dist_print(
            f"SM90 MXFP4 split bench: ranks={num_ranks} H={args.hidden} "
            f"I={args.intermediate_hidden} E={args.num_experts} "
            f"topk={args.num_topk} block_n={args.block_n} "
            f"attn_tp={args.attn_tp_size} attn_dp={num_ranks // args.attn_tp_size} "
            f"router={'bf16-e2e' if args.bf16_e2e else ('scalar-fused' if args.fuse_router else ('torch-tc+fused-topk' if args.torch_router_fused_topk else ('torch-tc' if args.torch_router else 'external')))} "
            f"tp_combine={'allgather' if args.tp_combine_allgather else ('reduce+broadcast' if args.tp_combine_broadcast else 'none')} "
            f"flush_l2={os.environ.get('DG_BENCH_FLUSH_L2_BYTES', 'default')}",
            once_in_node=True,
        )
        batches = args.batches[:1] if args.ncu_profile_only else args.batches
        for m_tokens in batches:
            _benchmark_case(
                args,
                m_tokens,
                rank_idx,
                num_ranks,
                group,
                attn_tp_group,
                buffer,
                transformed_l1,
                transformed_l2,
                router_weight,
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser(description="SM90 MXFP4 split MegaMoE benchmark")
    parser.add_argument("--ncu-profile-only", action="store_true")
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--attn-tp-size", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--block-n", type=int, choices=[128, 256], required=True)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--global-scales", action="store_true")
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--weight-seed", type=int, default=20260703)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--router-seed", type=int, default=20260805)
    parser.add_argument("--fuse-router", action="store_true")
    parser.add_argument("--torch-router", action="store_true")
    parser.add_argument("--torch-router-fused-topk", action="store_true")
    parser.add_argument("--router-repeat-tp", action="store_true")
    parser.add_argument("--bf16-e2e", action="store_true")
    parser.add_argument("--tp-combine-allgather", action="store_true")
    parser.add_argument("--tp-combine-broadcast", action="store_true")
    parser.add_argument("--router-renormalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-tests", type=int, default=20)
    args = parser.parse_args()
    if args.tp_combine_allgather and args.tp_combine_broadcast:
        parser.error("TP collective modes are mutually exclusive")
    if sum(map(int, (args.fuse_router, args.torch_router, args.torch_router_fused_topk, args.bf16_e2e))) > 1:
        parser.error("router mode flags are mutually exclusive")
    if args.tp_combine_allgather and any(m % args.attn_tp_size for m in args.batches):
        parser.error("all batches must be divisible by attn_tp_size for TP all-gather combine")
    return args


if __name__ == "__main__":
    parsed_args = _parse_args()
    if parsed_args.local_rank_idx is not None:
        _worker(parsed_args.local_rank_idx, parsed_args.num_processes, parsed_args)
    else:
        torch.multiprocessing.spawn(
            _worker,
            args=(parsed_args.num_processes, parsed_args),
            nprocs=parsed_args.num_processes,
            join=True,
        )
