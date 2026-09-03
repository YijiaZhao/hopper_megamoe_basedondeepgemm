"""CUDA Graph E2E benchmark/profile for the fixed H20 customer workload.

Shape: 8 GPUs, TP4/DP2/EP8, one real token per DP replica (global M=2),
E384/H3072/I1280/TopK8. The graph contains TP ReduceScatter, fused MXFP4
Router+Quant+TopK8, benchmark-only balanced metadata, split MegaMoE L1/L2,
and TP AllGather.

Run with torchrun directly for timing, or wrap torchrun with nsys and
--cuda-graph-trace=node for a single 8-GPU report. No cudaProfilerStart/Stop.
"""
import argparse
import os
import sys
import types

import torch
import torch.distributed as dist
from torch.cuda import nvtx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import deep_gemm
from bench_mxfp4_mega_moe_sm90 import _prepare_weights

WORLD = 8
TP = 4
PADDED_TOKENS_PER_TP_GROUP = 4
HIDDEN = 3072
INTERMEDIATE = 1280
EXPERTS = 384
TOPK = 8


def make_tp_group(rank):
    mine = None
    for first in range(0, WORLD, TP):
        candidate = dist.new_group(list(range(first, first + TP)))
        if first <= rank < first + TP:
            mine = candidate
    return mine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--routing", choices=("balanced-distinct", "balanced-same"),
                   default="balanced-distinct")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--flush-l2-mib", type=int, default=0,
                   help="Flush this many MiB per GPU before each replay; 0 keeps hot cache")
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == WORLD
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    tp_group = make_tp_group(rank)
    tp_rank = dist.get_rank(tp_group)
    dp_rank = rank // TP
    local_experts = EXPERTS // WORLD

    model = types.SimpleNamespace(
        weight_seed=20260703, weight_scale=0.05,
        hidden=HIDDEN, intermediate_hidden=INTERMEDIATE, block_n=128)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, EXPERTS, 1, TOPK, HIDDEN, INTERMEDIATE,
        mma_type="fp8xmxfp4", activation="swiglu")
    try:
        l1, l2 = _prepare_weights(model, local_experts, rank)
        torch.manual_seed(20260805)
        router_weight = (
            torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16)
            * 0.05).contiguous()
        torch.manual_seed(17000 + dp_rank * 1000003)
        real_token = torch.randn(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
        partial = torch.zeros(
            PADDED_TOKENS_PER_TP_GROUP, HIDDEN,
            device="cuda", dtype=torch.bfloat16)
        partial[0].copy_(real_token[0])
        partial_work = torch.empty_like(partial)
        local_hidden = torch.empty(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
        local_output = torch.empty_like(local_hidden)
        output = torch.empty_like(partial)
        stats = torch.zeros(local_experts, device="cuda", dtype=torch.int32)

        slots = torch.arange(TOPK, device="cuda", dtype=torch.int64)
        dp_offset = dp_rank * TOPK if args.routing == "balanced-distinct" else 0
        expert_offset = (dp_offset + slots * 7) % local_experts
        valid_idx = (slots * local_experts + expert_offset)[None, :]
        valid_weight = torch.full((1, TOPK), 1.0 / TOPK, device="cuda")
        invalid_idx = torch.full(
            (1, TOPK), -1, device="cuda", dtype=torch.int64)
        invalid_weight = torch.zeros((1, TOPK), device="cuda")
        flush = None
        if args.flush_l2_mib:
            flush = torch.empty(
                args.flush_l2_mib * 1024 * 1024 // 4,
                device="cuda", dtype=torch.int32)

        def body():
            nvtx.range_push(f"rank{rank}/input_copy_zero")
            partial_work.copy_(partial)
            stats.zero_()
            nvtx.range_pop()

            nvtx.range_push(f"rank{rank}/tp_reduce_scatter")
            dist.reduce_scatter_tensor(local_hidden, partial_work, group=tp_group)
            nvtx.range_pop()

            nvtx.range_push(f"rank{rank}/fused_router_quant_topk8")
            deep_gemm._C.mxfp4_router_quant_topk(
                local_hidden, router_weight,
                buffer.x[:1], buffer.x_sf[:1],
                buffer.topk_idx[:1], buffer.topk_weights[:1])
            nvtx.range_pop()

            nvtx.range_push(f"rank{rank}/balanced_metadata")
            if tp_rank == 0:
                buffer.topk_idx[:1].copy_(valid_idx)
                buffer.topk_weights[:1].copy_(valid_weight)
            else:
                buffer.topk_idx[:1].copy_(invalid_idx)
                buffer.topk_weights[:1].copy_(invalid_weight)
            nvtx.range_pop()

            nvtx.range_push(f"rank{rank}/megamoe")
            deep_gemm.mxfp4_mega_moe(
                local_output, l1, l2, buffer,
                cumulative_local_expert_recv_stats=stats,
                recipe=(128, 128, 128), activation_clamp=10.0,
                attn_tp_size=TP, attn_tp_group=tp_group,
                broadcast_output=False)
            nvtx.range_pop()

            nvtx.range_push(f"rank{rank}/tp_allgather")
            dist.all_gather_into_tensor(output, local_output, group=tp_group)
            nvtx.range_pop()

        # Materialize JIT kernels and NCCL communicators before capture.
        for _ in range(3):
            body()
        torch.cuda.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(capture_stream):
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(
                    graph, stream=capture_stream, capture_error_mode="relaxed"):
                body()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        dist.barrier()

        for _ in range(args.warmup):
            if flush is not None:
                flush.zero_()
            graph.replay()
            torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nvtx.range_push(f"rank{rank}/MEASURE_{args.iters}_GRAPH_REPLAYS")
        start.record()
        for i in range(args.iters):
            if flush is not None:
                flush.zero_()
                torch.cuda.synchronize()
                dist.barrier()
            nvtx.range_push(f"rank{rank}/replay_{i:02d}")
            graph.replay()
            nvtx.range_pop()
        end.record()
        end.synchronize()
        nvtx.range_pop()

        elapsed_us = start.elapsed_time(end) * 1000.0 / args.iters
        value = torch.tensor(elapsed_us, device="cuda", dtype=torch.float64)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(
                f"GRAPH_E2E routing={args.routing} warmup={args.warmup} "
                f"iters={args.iters} flush_l2_mib={args.flush_l2_mib} "
                f"mean_rank_us={value.item() / WORLD:.3f}", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
