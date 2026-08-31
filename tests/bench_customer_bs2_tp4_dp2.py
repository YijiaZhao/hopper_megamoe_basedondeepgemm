"""Benchmark TP ReduceScatter -> fused MegaMoE -> token AllGather."""
import argparse
import os
import sys
import types

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils.dist import init_dist
from bench_mxfp4_mega_moe_sm90 import _prepare_weights


def _make_tp_group(rank, world, tp_size):
    result = None
    for first in range(0, world, tp_size):
        ranks = list(range(first, first + tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            result = group
    assert result is not None
    return result


def _worker(local_rank, world, args):
    rank, num_ranks, ep_group = init_dist(local_rank, world)
    assert num_ranks == world and get_arch_major() == 9
    tp_group = _make_tp_group(rank, world, args.tp_size)
    tp_rank = dist.get_rank(tp_group)
    dp_rank = rank // args.tp_size
    model = types.SimpleNamespace(
        weight_seed=20260703, weight_scale=0.05,
        hidden=3072, intermediate_hidden=1280, block_n=128)
    max_local_tokens = max(args.batches) // args.tp_size
    sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        ep_group, 384, max_local_tokens, 8, 3072, 1280,
        mma_type="fp8xmxfp4", activation="swiglu")
    try:
        l1_weights, l2_weights = _prepare_weights(
            model, 384 // num_ranks, rank)
        torch.manual_seed(20260805)
        router_weight = (
            torch.randn(384, 3072, device="cuda", dtype=torch.bfloat16)
            * 0.05).contiguous()
        stats = torch.zeros(384 // num_ranks, device="cuda", dtype=torch.int32)
        if rank == 0:
            precision = "QoQ" if os.environ.get("DG_W4A8_INT", "0") != "0" else "scaled-MXFP4"
            print(f"CUSTOM bs_per_dp=1 isl=4 tokens_per_tp_group=4 global_tokens=8 E384 H3072 I1280 TopK8 precision={precision} TP={args.tp_size} DP={world // args.tp_size}")

        for num_tokens in args.batches:
            assert num_tokens % args.tp_size == 0
            local_tokens = num_tokens // args.tp_size
            torch.manual_seed(17000 + num_tokens * 1009 + dp_rank * 1000003)
            full_hidden = torch.randn(
                num_tokens, 3072, device="cuda", dtype=torch.bfloat16)
            partial_template = (
                full_hidden.float() / args.tp_size).to(torch.bfloat16).contiguous()
            partial_work = torch.empty_like(partial_template)
            local_hidden = torch.empty(
                local_tokens, 3072, device="cuda", dtype=torch.bfloat16)
            local_output = torch.empty_like(local_hidden)
            output = torch.empty_like(full_hidden)
            router_logits_e2e = torch.empty(
                (local_tokens, 384), device="cuda", dtype=torch.bfloat16)
            global_token = (
                torch.arange(local_tokens, device="cuda", dtype=torch.int64)
                + dp_rank * num_tokens + tp_rank * local_tokens)
            route_slot = torch.arange(8, device="cuda", dtype=torch.int64)
            # Each DP replica has BS1 x ISL4 = 4 real tokens. After TP4 RS,
            # every physical rank owns one token. Give every token one TopK
            # assignment on each EP rank, yielding exactly 8 assignments/rank.
            target_ep_rank = route_slot
            expert_offset = (global_token[:, None] * 8 + route_slot[None, :] * 7) % (384 // world)
            balanced_idx = (
                target_ep_rank[None, :] * (384 // world)
                + expert_offset
            ).contiguous()
            balanced_weight = torch.full(
                (local_tokens, 8), 1.0 / 8.0, device="cuda",
                dtype=torch.float32)
            valid_group_tokens = args.valid_tokens_per_group or num_tokens
            assert 0 < valid_group_tokens <= num_tokens
            valid_local_tokens = max(
                0, min(local_tokens,
                       valid_group_tokens - tp_rank * local_tokens))

            def run():
                partial_work.copy_(partial_template)
                stats.zero_()
                if (not args.balanced_routing and
                        valid_group_tokens == num_tokens):
                    deep_gemm.mxfp4_mega_moe_from_tp_partial_bf16(
                        output, partial_work, local_hidden, local_output,
                        router_weight, l1_weights, l2_weights, sym_buffer,
                        cumulative_local_expert_recv_stats=stats,
                        recipe=(128, 128, 128), activation_clamp=10.0,
                        attn_tp_size=args.tp_size, attn_tp_group=tp_group)
                else:
                    dist.reduce_scatter_tensor(
                        local_hidden, partial_work, group=tp_group)
                    if os.environ.get("DG_W4A8_INT", "0") != "0":
                        deep_gemm._C.qoq_router_quant_topk(
                            local_hidden, router_weight, router_logits_e2e,
                            sym_buffer.x[:local_tokens],
                            sym_buffer.x_sf[:local_tokens],
                            sym_buffer.topk_idx[:local_tokens],
                            sym_buffer.topk_weights[:local_tokens])
                    else:
                        deep_gemm._C.mxfp4_router_quant_topk_split(
                            local_hidden, router_weight, router_logits_e2e,
                            sym_buffer.x[:local_tokens],
                            sym_buffer.x_sf[:local_tokens],
                            sym_buffer.topk_idx[:local_tokens],
                            sym_buffer.topk_weights[:local_tokens])
                    if args.balanced_routing:
                        sym_buffer.topk_idx[:local_tokens].copy_(balanced_idx)
                        sym_buffer.topk_weights[:local_tokens].copy_(balanced_weight)
                    if valid_local_tokens < local_tokens:
                        sym_buffer.topk_idx[valid_local_tokens:local_tokens].fill_(-1)
                        sym_buffer.topk_weights[valid_local_tokens:local_tokens].zero_()
                    deep_gemm.mxfp4_mega_moe(
                        local_output, l1_weights, l2_weights, sym_buffer,
                        cumulative_local_expert_recv_stats=stats,
                        recipe=(128, 128, 128), activation_clamp=10.0,
                        attn_tp_size=args.tp_size, attn_tp_group=tp_group,
                        broadcast_output=False)
                    dist.all_gather_into_tensor(
                        output, local_output, group=tp_group)

            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()
            dist.barrier(group=ep_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                run()
            end.record()
            torch.cuda.synchronize()
            elapsed_us = start.elapsed_time(end) * 1000.0 / args.iters
            measured = torch.tensor(elapsed_us, device="cuda", dtype=torch.float64)
            dist.all_reduce(measured, op=dist.ReduceOp.MAX, group=ep_group)
            if rank == 0:
                print(
                    f"RESULT global_tokens=8 M_per_tp_group={num_tokens} max_rank_us={measured.item():.3f}",
                    flush=True)
    finally:
        sym_buffer.destroy()
        dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, choices=[4], default=4)
    parser.add_argument("--batches", nargs="+", type=int,
                        default=[4])
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--balanced-routing", action="store_true", default=True)
    parser.add_argument("--valid-tokens-per-group", type=int, default=4,
                        help="Mask padded tokens after the fused frontend; balanced-routing only")
    args = parser.parse_args()
    if args.num_processes % args.tp_size:
        parser.error("num-processes must be divisible by tp-size")
    if any(m % args.tp_size for m in args.batches):
        parser.error("every padded batch must be divisible by tp-size")
    return args


if __name__ == "__main__":
    parsed = _parse_args()
    torch.multiprocessing.spawn(
        _worker, args=(parsed.num_processes, parsed),
        nprocs=parsed.num_processes, join=True)
