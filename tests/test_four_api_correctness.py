"""Exact-quantized-reference validation for the four H20 MegaMoE APIs.

The test intentionally calls MXFP4 and QoQ in the same process, so it also
checks that the explicit APIs do not depend on process-global DG_W4A8_INT.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deep_gemm
from deep_gemm.quantization_mxfp4 import (
    dequantize_mxfp4_to_fp32 as dequantize_mxfp4_split,
    dequantize_qoq_int4,
    quantize_to_mxfp4 as quantize_to_mxfp4_split,
    quantize_to_qoq_int4,
)
from deep_gemm.quantization_mxfp4_fused import (
    dequantize_mxfp4_kernel_exact_fp32,
    mxfp4_row_reference_exponent,
    quantize_to_mxfp4 as quantize_to_mxfp4_fused,
)
from deep_gemm.quantization_qoq_fused import (
    dequantize_qoq_to_fp32,
    int8_bytes_to_float,
    per_token_cast_to_int8,
    quantize_to_qoq,
)
from deep_gemm.utils import per_token_cast_to_fp8


def _silu(x):
    return x * torch.sigmoid(x)


def _quantize_intermediate(x, qoq):
    groups = x.view(-1, 64)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    if qoq:
        scale = (amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
        return ((groups / scale).round().clamp(-127, 127) * scale).view_as(x)
    scaled_amax = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.exp2(torch.ceil(torch.log2(scaled_amax)))
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).view_as(x)


def _metrics(y, ref, group):
    diff = y.float() - ref
    max_abs = diff.abs().max()
    sum_abs = diff.abs().sum()
    cos = torch.nn.functional.cosine_similarity(y.float(), ref, dim=-1)
    cos_min = cos.min()
    cos_sum = cos.sum()
    y_norm_sq = y.float().square().sum()
    ref_norm_sq = ref.square().sum()
    finite = torch.tensor(int(torch.isfinite(y).all()), device=y.device, dtype=torch.int32)
    for tensor, op in (
        (max_abs, dist.ReduceOp.MAX),
        (sum_abs, dist.ReduceOp.SUM),
        (cos_min, dist.ReduceOp.MIN),
        (cos_sum, dist.ReduceOp.SUM),
        (y_norm_sq, dist.ReduceOp.SUM),
        (ref_norm_sq, dist.ReduceOp.SUM),
        (finite, dist.ReduceOp.MIN),
    ):
        dist.all_reduce(tensor, op=op, group=group)
    world = dist.get_world_size(group)
    return {
        "finite": bool(finite.item()),
        "max_abs": max_abs.item(),
        "mean_abs": sum_abs.item() / (y.numel() * world),
        "cos_min": cos_min.item(),
        "cos_mean": cos_sum.item() / (y.size(0) * world),
        "norm_ratio": torch.sqrt(y_norm_sq / ref_norm_sq.clamp_min(1e-30)).item(),
    }


def _run(api, args, rank, group):
    world = dist.get_world_size(group)
    local_experts = args.experts // world
    is_qoq = api.startswith("qoq")
    is_fused = api.endswith("fused")

    torch.manual_seed(args.seed + rank * 1000003)
    x_bf = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    w1_bf = torch.randn(
        local_experts, 2 * args.intermediate, args.hidden,
        device="cuda", dtype=torch.bfloat16) * args.weight_scale
    w2_bf = torch.randn(
        local_experts, args.hidden, args.intermediate,
        device="cuda", dtype=torch.bfloat16) * args.weight_scale

    # Balanced routing: every source token has one route owned by every rank.
    slots = torch.arange(args.topk, device="cuda", dtype=torch.int64)
    assert args.topk == world
    token_ids = rank * args.tokens + torch.arange(args.tokens, device="cuda")
    offsets = (token_ids[:, None] * args.topk + slots[None, :] * 7) % local_experts
    topk_idx = slots[None, :] * local_experts + offsets
    topk_weights = torch.full(
        (args.tokens, args.topk), 1.0 / args.topk,
        device="cuda", dtype=torch.float32)

    if is_qoq:
        xq, xs = per_token_cast_to_int8(x_bf, gran_k=128)
        if is_fused:
            q1, q2 = quantize_to_qoq(w1_bf), quantize_to_qoq(w2_bf)
            d1, d2 = dequantize_qoq_to_fp32(*q1), dequantize_qoq_to_fp32(*q2)
            weights = deep_gemm.transform_qoq_weights_for_mega_moe_fused(q1, q2)
        else:
            q1, q2 = quantize_to_qoq_int4(w1_bf), quantize_to_qoq_int4(w2_bf)
            d1, d2 = dequantize_qoq_int4(*q1), dequantize_qoq_int4(*q2)
            weights = deep_gemm.transform_qoq_int4_weights_for_mega_moe_sm90(q1, q2, block_n=128)
        x_ref_local = int8_bytes_to_float(xq) * xs[:, :1]
    else:
        xq, xs = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)
        if is_fused:
            q1, q2 = quantize_to_mxfp4_fused(w1_bf), quantize_to_mxfp4_fused(w2_bf)
            d1 = dequantize_mxfp4_kernel_exact_fp32(q1[0], q1[1], mxfp4_row_reference_exponent(q1[1]))
            d2 = dequantize_mxfp4_kernel_exact_fp32(q2[0], q2[1], mxfp4_row_reference_exponent(q2[1]))
            weights = deep_gemm.transform_mxfp4_weights_for_mega_moe_fused(q1, q2)
        else:
            q1, q2 = quantize_to_mxfp4_split(w1_bf), quantize_to_mxfp4_split(w2_bf)
            d1, d2 = dequantize_mxfp4_split(*q1), dequantize_mxfp4_split(*q2)
            weights = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(q1, q2, block_n=128)
        x_ref_local = (
            xq.float().view(args.tokens, args.hidden // 128, 128)
            * xs.float().unsqueeze(-1)
        ).view(args.tokens, args.hidden)

    del w1_bf, w2_bf
    torch.cuda.empty_cache()

    if is_fused:
        buffer = deep_gemm.get_fused_symm_buffer_for_mega_moe(
            group, args.experts, args.tokens, args.topk, args.hidden, args.intermediate)
        kernel = getattr(deep_gemm, api)
    else:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, args.experts, args.tokens, args.topk, args.hidden, args.intermediate,
            mma_type="fp8xmxfp4", activation="swiglu")
        kernel = getattr(deep_gemm, api)

    try:
        buffer.x[:args.tokens].copy_(xq)
        buffer.x_sf[:args.tokens].copy_(xs)
        buffer.topk_idx[:args.tokens].copy_(topk_idx)
        buffer.topk_weights[:args.tokens].copy_(topk_weights)
        y = torch.zeros(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
        if is_fused:
            kernel(y, *weights, buffer, activation_clamp=args.activation_clamp)
        else:
            kernel(
                y, *weights, buffer, recipe=(128, 128, 128),
                activation_clamp=args.activation_clamp)
        torch.cuda.synchronize()
        dist.barrier(group=group)

        global_tokens = args.tokens * world
        x_all = torch.empty(global_tokens, args.hidden, device="cuda", dtype=torch.float32)
        idx_all = torch.empty(global_tokens, args.topk, device="cuda", dtype=torch.int64)
        weight_all = torch.empty(global_tokens, args.topk, device="cuda", dtype=torch.float32)
        dist.all_gather_into_tensor(x_all, x_ref_local.contiguous(), group=group)
        dist.all_gather_into_tensor(idx_all, topk_idx.contiguous(), group=group)
        dist.all_gather_into_tensor(weight_all, topk_weights.contiguous(), group=group)

        routes = torch.zeros(
            global_tokens, args.topk, args.hidden, device="cuda", dtype=torch.float32)
        for token in range(global_tokens):
            for slot in range(args.topk):
                expert = int(idx_all[token, slot])
                if expert // local_experts != rank:
                    continue
                local_expert = expert % local_experts
                l1 = d1[local_expert].float() @ x_all[token]
                gate, up = l1[:args.intermediate], l1[args.intermediate:]
                gate = gate.clamp(max=args.activation_clamp)
                up = up.clamp(-args.activation_clamp, args.activation_clamp)
                mid = _silu(gate) * up * float(weight_all[token, slot])
                mid = _quantize_intermediate(mid, is_qoq)
                routes[token, slot] = (d2[local_expert].float() @ mid).to(torch.bfloat16).float()
        dist.all_reduce(routes, op=dist.ReduceOp.SUM, group=group)
        ref_all = routes.sum(dim=1).to(torch.bfloat16).float()
        ref = ref_all[rank * args.tokens:(rank + 1) * args.tokens]
        metrics = _metrics(y, ref, group)
        if rank == 0:
            print(
                f"RESULT api={api} finite={int(metrics['finite'])} "
                f"max_abs={metrics['max_abs']:.8g} mean_abs={metrics['mean_abs']:.8g} "
                f"cos_min={metrics['cos_min']:.10f} cos_mean={metrics['cos_mean']:.10f} "
                f"norm_ratio={metrics['norm_ratio']:.10f}", flush=True)
        assert metrics["finite"], api
        assert metrics["cos_min"] >= args.cosine_min, (api, metrics)
        assert args.norm_ratio_min <= metrics["norm_ratio"] <= args.norm_ratio_max, (api, metrics)
    finally:
        buffer.destroy()
        dist.barrier(group=group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apis", nargs="+", default=[
        "mxfp4_mega_moe_split", "qoq_mega_moe_split",
        "mxfp4_mega_moe_fused", "qoq_mega_moe_fused"])
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--cosine-min", type=float, default=0.99)
    parser.add_argument("--norm-ratio-min", type=float, default=0.97)
    parser.add_argument("--norm-ratio-max", type=float, default=1.03)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    try:
        assert dist.get_world_size() == 8
        for api in args.apis:
            _run(api, args, rank, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
