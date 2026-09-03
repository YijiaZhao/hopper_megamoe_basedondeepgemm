"""Nsight Systems timeline driver for the four explicit H20 MegaMoE APIs.

Matrix dimensions:
  scope:     e2e, mega
  backend:   split, fused
  quant:     mxfp4, qoq
  global M:  2, 8, 16

E2E captures TP4 ReduceScatter, router/activation-quant/TopK, MegaMoE, and TP4
AllGather.  Only frontend+MegaMoE is placed in a CUDA graph because NCCL graph
capture is not reliable in the delivery environment.  Mega-only captures the
explicit MegaMoE API in a CUDA graph.  L2 eviction is outside NVTX ranges.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.cuda import nvtx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deep_gemm
from deep_gemm.quantization_mxfp4 import (
    quantize_to_mxfp4 as quantize_to_mxfp4_split,
    quantize_to_qoq_int4,
)
from deep_gemm.quantization_mxfp4_fused import quantize_to_mxfp4 as quantize_to_mxfp4_fused
from deep_gemm.quantization_qoq_fused import quantize_to_qoq
from deep_gemm.testing.bench import flush_l2_cache
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.quantization_qoq_fused import per_token_cast_to_int8

WORLD = 8
TP = 4
HIDDEN = 3072
INTERMEDIATE = 1280
EXPERTS = 384
TOPK = 8
WARMUP = 2
ITERS = 8


def make_tp_group(rank):
    mine = None
    for first in range(0, WORLD, TP):
        candidate = dist.new_group(list(range(first, first + TP)))
        if first <= rank < first + TP:
            mine = candidate
    return mine


def prepare_backend(args, rank, local_rows, group):
    local_experts = EXPERTS // WORLD
    torch.manual_seed(20260703 + rank * 1000003)
    w1 = torch.randn(
        local_experts, 2 * INTERMEDIATE, HIDDEN,
        device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(
        local_experts, HIDDEN, INTERMEDIATE,
        device="cuda", dtype=torch.bfloat16) * 0.05

    if args.backend == "split":
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, EXPERTS, local_rows, TOPK, HIDDEN, INTERMEDIATE,
            mma_type="fp8xmxfp4", activation="swiglu")
        if args.quant == "mxfp4":
            weights = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(
                quantize_to_mxfp4_split(w1), quantize_to_mxfp4_split(w2), block_n=128)
            kernel = deep_gemm.mxfp4_mega_moe_split
        else:
            weights = deep_gemm.transform_qoq_int4_weights_for_mega_moe_sm90(
                quantize_to_qoq_int4(w1), quantize_to_qoq_int4(w2), block_n=128)
            kernel = deep_gemm.qoq_mega_moe_split

        def launch(y):
            kernel(y, *weights, buffer, cumulative_local_expert_recv_stats=None,
                   recipe=(128, 128, 128), activation_clamp=10.0)
    else:
        buffer = deep_gemm.get_fused_symm_buffer_for_mega_moe(
            group, EXPERTS, local_rows, TOPK, HIDDEN, INTERMEDIATE)
        if args.quant == "mxfp4":
            weights = deep_gemm.transform_mxfp4_weights_for_mega_moe_fused(
                quantize_to_mxfp4_fused(w1), quantize_to_mxfp4_fused(w2))
            kernel = deep_gemm.mxfp4_mega_moe_fused
        else:
            weights = deep_gemm.transform_qoq_weights_for_mega_moe_fused(
                quantize_to_qoq(w1), quantize_to_qoq(w2))
            kernel = deep_gemm.qoq_mega_moe_fused

        def launch(y):
            kernel(y, *weights, buffer, cumulative_local_expert_recv_stats=None,
                   activation_clamp=10.0)
    return buffer, launch


def launch_frontend(quant, x, router_weight, logits, buffer, rows):
    if quant == "mxfp4":
        deep_gemm._C.mxfp4_router_quant_topk(
            x, router_weight, buffer.x[:rows], buffer.x_sf[:rows],
            buffer.topk_idx[:rows], buffer.topk_weights[:rows])
    else:
        deep_gemm._C.qoq_router_quant_topk(
            x, router_weight, logits, buffer.x[:rows], buffer.x_sf[:rows],
            buffer.topk_idx[:rows], buffer.topk_weights[:rows])


def run_e2e(args, rank, tp_group, group):
    dp_rank = rank // TP
    unique_per_dp = args.global_tokens // 2
    local_rows = max(1, args.global_tokens // WORLD)
    padded_rows = local_rows * TP
    buffer, launch_moe = prepare_backend(args, rank, local_rows, group)
    try:
        torch.manual_seed(20260805)
        router_weight = (
            torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05
        ).contiguous()
        logits = torch.empty(local_rows, EXPERTS, device="cuda", dtype=torch.bfloat16)
        partials = []
        for input_id in range(WARMUP + ITERS):
            partial = torch.zeros(padded_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
            for token_id in range(unique_per_dp):
                owner = token_id % TP
                row = token_id // TP
                torch.manual_seed(17000 + dp_rank * 1000003 + input_id * 7919 + token_id * 101)
                partial[owner * local_rows + row].copy_(
                    torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16))
            partials.append(partial)

        work = torch.empty_like(partials[0])
        x = torch.empty(local_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        output = torch.empty(padded_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)

        def graph_body():
            launch_frontend(args.quant, x, router_weight, logits, buffer, local_rows)
            launch_moe(y)

        work.copy_(partials[0])
        dist.reduce_scatter_tensor(x, work, group=tp_group)
        graph_body()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        graph_stream = torch.cuda.Stream()
        graph_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_stream, capture_error_mode="relaxed"):
            graph_body()
        graph_stream.synchronize()
        dist.barrier(group=group)

        def replay(input_id, annotate):
            flush_l2_cache()
            torch.cuda.synchronize()
            work.copy_(partials[input_id])
            if annotate:
                nvtx.range_push(f"rank{rank}/tp_reduce_scatter")
            dist.reduce_scatter_tensor(x, work, group=tp_group)
            if annotate:
                nvtx.range_pop()
                nvtx.range_push(
                    f"rank{rank}/frontend_megamoe_graph/{args.backend}/{args.quant}/M{args.global_tokens}")
            graph.replay()
            if annotate:
                nvtx.range_pop()
                nvtx.range_push(f"rank{rank}/tp_allgather")
            dist.all_gather_into_tensor(output, y, group=tp_group)
            if annotate:
                nvtx.range_pop()

        for i in range(WARMUP):
            replay(i, False)
            torch.cuda.synchronize()
            dist.barrier(group=group)
        nvtx.range_push(
            f"rank{rank}/MEASURE_{ITERS}/e2e/{args.backend}/{args.quant}/M{args.global_tokens}")
        for i in range(ITERS):
            dist.barrier(group=group)
            nvtx.range_push(f"rank{rank}/iter_{i:02d}")
            replay(i + WARMUP, True)
            nvtx.range_pop()
            torch.cuda.synchronize()
            dist.barrier(group=group)
        nvtx.range_pop()
        torch.cuda.synchronize()
        dist.barrier(group=group)
    finally:
        buffer.destroy()


def local_tokens(global_tokens, rank):
    if global_tokens == 2:
        return 1 if rank in (0, 4) else 0
    return global_tokens // WORLD


def run_mega(args, rank, group):
    active_rows = local_tokens(args.global_tokens, rank)
    local_rows = max(1, args.global_tokens // WORLD)
    local_experts = EXPERTS // WORLD
    buffer, launch_moe = prepare_backend(args, rank, local_rows, group)
    try:
        torch.manual_seed(17000 + rank * 1000003 + args.global_tokens)
        x = torch.randn(local_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        if args.quant == "mxfp4":
            xq, xs = per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)
        else:
            xq, xs = per_token_cast_to_int8(x, gran_k=128)
        buffer.x[:local_rows].copy_(xq)
        buffer.x_sf[:local_rows].copy_(xs)
        buffer.topk_idx[:local_rows].fill_(-1)
        buffer.topk_weights[:local_rows].zero_()
        if active_rows:
            token = torch.arange(active_rows, device="cuda", dtype=torch.int64)[:, None]
            slot = torch.arange(TOPK, device="cuda", dtype=torch.int64)[None, :]
            global_token = rank * local_rows + token
            idx = slot * local_experts + ((global_token + slot * 7) % local_experts)
            buffer.topk_idx[:active_rows].copy_(idx)
            buffer.topk_weights[:active_rows].fill_(1.0 / TOPK)
        y = torch.empty(local_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)

        launch_moe(y)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            launch_moe(y)
        torch.cuda.synchronize()
        dist.barrier(group=group)

        def replay(annotate):
            flush_l2_cache()
            torch.cuda.synchronize()
            if annotate:
                nvtx.range_push(
                    f"rank{rank}/megamoe_graph/{args.backend}/{args.quant}/M{args.global_tokens}")
            graph.replay()
            if annotate:
                nvtx.range_pop()

        for _ in range(WARMUP):
            replay(False)
            torch.cuda.synchronize()
            dist.barrier(group=group)
        nvtx.range_push(
            f"rank{rank}/MEASURE_{ITERS}/mega/{args.backend}/{args.quant}/M{args.global_tokens}")
        for i in range(ITERS):
            dist.barrier(group=group)
            nvtx.range_push(f"rank{rank}/iter_{i:02d}")
            replay(True)
            nvtx.range_pop()
            torch.cuda.synchronize()
        nvtx.range_pop()
        dist.barrier(group=group)
    finally:
        buffer.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("e2e", "mega"), required=True)
    parser.add_argument("--backend", choices=("split", "fused"), required=True)
    parser.add_argument("--quant", choices=("mxfp4", "qoq"), required=True)
    parser.add_argument("--global-tokens", type=int, choices=(2, 8, 16), required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    tp_group = make_tp_group(rank)
    try:
        if args.scope == "e2e":
            run_e2e(args, rank, tp_group, dist.group.WORLD)
        else:
            run_mega(args, rank, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
