"""One-process smoke test for all four explicit H20 MegaMoE APIs."""
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deep_gemm
from deep_gemm.quantization_mxfp4 import (
    quantize_to_mxfp4 as quantize_to_mxfp4_split,
    quantize_to_qoq_int4,
)
from deep_gemm.quantization_mxfp4_fused import quantize_to_mxfp4 as quantize_to_mxfp4_fused
from deep_gemm.quantization_qoq_fused import quantize_to_qoq, per_token_cast_to_int8
from deep_gemm.utils import per_token_cast_to_fp8


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    experts, hidden, intermediate, topk, m = 384, 3072, 1280, 8, 1
    local_experts = experts // dist.get_world_size()
    torch.manual_seed(100 + rank)
    x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(local_experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(local_experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.05
    slots = torch.arange(topk, device="cuda")
    topk_idx = (slots * local_experts + ((rank + slots * 7) % local_experts))[None, :].to(torch.int64)
    topk_weights = torch.full((m, topk), 1 / topk, device="cuda", dtype=torch.float32)

    split_buffer = None
    fused_buffer = None
    try:
        split_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD, experts, m, topk, hidden, intermediate,
            mma_type="fp8xmxfp4", activation="swiglu")
        fused_buffer = deep_gemm.get_fused_symm_buffer_for_mega_moe(
            dist.group.WORLD, experts, m, topk, hidden, intermediate)

        # Prepare both formats once; all four explicit APIs are then called in
        # the same process without changing DG_W4A8_INT.
        mx_split = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(
            quantize_to_mxfp4_split(w1), quantize_to_mxfp4_split(w2), block_n=128)
        qoq_split = deep_gemm.transform_qoq_int4_weights_for_mega_moe_sm90(
            quantize_to_qoq_int4(w1), quantize_to_qoq_int4(w2), block_n=128)
        mx_fused = deep_gemm.transform_mxfp4_weights_for_mega_moe_fused(
            quantize_to_mxfp4_fused(w1), quantize_to_mxfp4_fused(w2))
        qoq_fused = deep_gemm.transform_qoq_weights_for_mega_moe_fused(
            quantize_to_qoq(w1), quantize_to_qoq(w2))

        x_fp8, x_fp8_sf = per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)
        x_int8, x_int8_sf = per_token_cast_to_int8(x, gran_k=128)

        cases = (
            ("mxfp4_split", split_buffer, mx_split, x_fp8, x_fp8_sf, deep_gemm.mxfp4_mega_moe_split),
            ("qoq_split", split_buffer, qoq_split, x_int8, x_int8_sf, deep_gemm.qoq_mega_moe_split),
            ("mxfp4_fused", fused_buffer, mx_fused, x_fp8, x_fp8_sf, deep_gemm.mxfp4_mega_moe_fused),
            ("qoq_fused", fused_buffer, qoq_fused, x_int8, x_int8_sf, deep_gemm.qoq_mega_moe_fused),
        )
        for name, buffer, weights, xq, xs, kernel in cases:
            buffer.x[:m].copy_(xq)
            buffer.x_sf[:m].copy_(xs)
            buffer.topk_idx[:m].copy_(topk_idx)
            buffer.topk_weights[:m].copy_(topk_weights)
            y = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
            if name.endswith("_split"):
                kernel(y, *weights, buffer, recipe=(128, 128, 128))
            else:
                kernel(y, *weights, buffer)
            torch.cuda.synchronize()
            assert torch.isfinite(y).all(), name
            if rank == 0:
                print(name, "finite", y.abs().mean().item(), flush=True)
            dist.barrier()
    finally:
        if split_buffer is not None:
            split_buffer.destroy()
        if fused_buffer is not None:
            fused_buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
