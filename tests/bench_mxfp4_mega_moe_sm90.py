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
from deep_gemm.quantization_mxfp4 import quantize_to_mxfp4
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
    l1_packed, l1_scale = quantize_to_mxfp4(l1_bf, group_size=32)
    del l1_bf

    l2_bf = torch.randn(
        (num_local_experts, args.hidden, args.intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * args.weight_scale
    l2_packed, l2_scale = quantize_to_mxfp4(l2_bf, group_size=32)
    del l2_bf

    transformed = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(
        (l1_packed, l1_scale),
        (l2_packed, l2_scale),
        block_n=args.block_n,
    )
    del l1_packed, l1_scale, l2_packed, l2_scale
    torch.cuda.empty_cache()
    return transformed


def _benchmark_case(
    args: argparse.Namespace,
    m_tokens: int,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer,
    transformed_l1,
    transformed_l2,
):
    torch.manual_seed(args.input_seed + m_tokens * 1009 + rank_idx * 1000003)
    x_bf = torch.randn(
        (m_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn(
        (m_tokens, args.num_experts), dtype=torch.float32, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, args.num_topk, dim=-1, largest=True, sorted=False)
    if os.environ.get('DG_BALANCED_ROUTING', '0') != '0':
        _p = ((torch.arange(m_tokens, device='cuda', dtype=torch.int64)
               + rank_idx * m_tokens) * args.num_topk).unsqueeze(1) + torch.arange(
            args.num_topk, device='cuda', dtype=torch.int64).unsqueeze(0)
        topk_idx = (_p * 101) % args.num_experts
        topk_weights = scores.gather(1, topk_idx)
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
    y = torch.empty((m_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    if args.global_scales:
        l1_global_scales = torch.linspace(
            0.73, 1.37, num_local_experts, dtype=torch.float32, device="cuda")
        l2_global_scales = torch.linspace(
            1.31, 0.67, num_local_experts, dtype=torch.float32, device="cuda")
    else:
        l1_global_scales = None
        l2_global_scales = None

    def run():
        buffer.x[:m_tokens].copy_(x_fp8)
        buffer.x_sf[:m_tokens].copy_(x_sf)
        buffer.topk_idx[:m_tokens].copy_(topk_idx)
        buffer.topk_weights[:m_tokens].copy_(topk_weights)
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
        dist.barrier(group=group)

        dist_print(
            f"SM90 MXFP4 split bench: ranks={num_ranks} H={args.hidden} "
            f"I={args.intermediate_hidden} E={args.num_experts} "
            f"topk={args.num_topk} block_n={args.block_n} "
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
                buffer,
                transformed_l1,
                transformed_l2,
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-tests", type=int, default=20)
    return parser.parse_args()


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
