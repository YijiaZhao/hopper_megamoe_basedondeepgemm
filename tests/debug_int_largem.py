"""Large-M smoke for the W4A8-int non-swapAB path.

Single process, all M tokens routed to local expert 0 with weight 1.0 so the
reference is a pair of plain matmuls (vectorized, no per-route python loop).

Run inside dg_dev:
  DG_W4A8_INT=1 DG_W4A8_INT_L2=1 python3 tests/debug_int_largem.py [M ...]
"""

import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from tests.debug_int_l2_instrument import (
    quantize_to_int4, dequant_int4, transform_int4, per_token_int8, _silu, q8_ref,
    _int4_scale_tm)
from deep_gemm.mega import _interleave_l1_weights as _il1

PRE = os.environ.get("DG_W4A8_INT_PRE", "0") == "1"
QOQ = os.environ.get("DG_W4A8_INT_QOQ", "0") == "1"
ZP = os.environ.get("DG_W4A8_INT_QOQ_ZP", "0") == "1"
PRELUT = os.environ.get("DG_W4A8_INT_QOQ_ZP_PRELUT", "0") == "1"
SHIFTXOR = os.environ.get("DG_W4A8_INT_QOQ_ZP_SHIFTXOR", "0") == "1"


def quantize_qoq(weight, group=128):
    # Level 1: per-row s1 (bf16-rounded so the kernel's masked read is exact;
    # +-120 so the s2 fold cannot overflow int8: 7*ceil(120/7) <= 126).
    w = weight.float()
    s1 = (w.abs().amax(dim=-1, keepdim=True) / 120.0).clamp_min(1e-30)
    s1 = s1.to(torch.bfloat16).to(torch.float32)
    w8 = (w / s1).round().clamp(-112 if ZP else -120, 112 if ZP else 120)
    E, N, K = w8.shape
    w8g = w8.view(E, N, K // group, group)
    s1w = (s1.view(torch.int32) & -65536).expand(E, N, K // group)
    if ZP:
        # Level 2 asymmetric: uint4 + zero point, integer s2.
        mn = w8g.amin(dim=-1, keepdim=True)
        mx = w8g.amax(dim=-1, keepdim=True)
        s2 = ((mx - mn) / 15.0).ceil().clamp(1, 127)
        z = (-mn / s2).round().clamp(0, 15)
        w4 = (w8g / s2 + z).round().clamp(0, 15)
        dec8 = ((w4 - z) * s2).view(E, N, K)
        assert dec8.abs().max().item() <= 127
        # Prepack byte 1: under PRELUT the kernel looks up a prestored smem
        # LUT row [s2][z] and under SHIFTXOR z is subtracted per byte (and the
        # smem-decode LUT built from it), so store the RAW zero point z;
        # otherwise store the negated offset nz = (-z*s2) mod 256 the
        # arithmetic LUT build needs.
        if PRELUT or SHIFTXOR:
            if PRELUT:
                assert s2.max().item() <= 31, "s2 exceeds prestored ZP LUT rows"
            b1 = z.to(torch.int32)
        else:
            b1 = (z * s2).to(torch.int32).neg().remainder(256)
        plane = (s1w | (b1.squeeze(-1) << 8)
                     | s2.squeeze(-1).to(torch.int32)).contiguous()
    else:
        amax8 = w8g.abs().amax(dim=-1, keepdim=True)
        s2 = (amax8 / 7.0).ceil().clamp(1, 127)
        w4 = (w8g / s2).round().clamp(-7, 7)
        dec8 = (w4 * s2).view(E, N, K)
        assert dec8.abs().max().item() <= 126
        plane = (s1w | s2.squeeze(-1).to(torch.int32)).contiguous()
    dequant = dec8 * s1
    w4i = w4.view(E, N, K).to(torch.int8)
    return dec8.to(torch.int8), plane, dequant, w4i


def _qoq_plane_tm(plane_i32, block_n=128):
    E, N, Kb = plane_i32.shape
    b = plane_i32.contiguous().view(torch.uint8).view(E, N, Kb, 4)
    return (b.view(E, N // block_n, block_n, Kb, 4).permute(0, 1, 3, 2, 4).contiguous())


def _pack_w4(w4i):
    E, N, K = w4i.shape
    nib = (w4i & 0x0F).to(torch.uint8).view(E, N, K // 8, 8)
    return (nib[..., 4:8] | (nib[..., 0:4] << 4)).view(E, N, K // 2).contiguous()


def decode_int4_to_int8(packed, N, K):
    E = packed.shape[0]
    b = packed.view(E, N, K // 8, 4)
    def s4(x):
        x = x.to(torch.int16); return torch.where(x >= 8, x - 16, x)
    q = torch.empty(E, N, K // 8, 8, dtype=torch.int16, device=packed.device)
    q[..., 0:4] = s4(b >> 4)
    q[..., 4:8] = s4(b & 0x0F)
    return q.to(torch.int8).view(E, N, K)


def transform_int4_pre(packed, scale, is_l1):
    if is_l1:
        packed, scale = _il1((packed, scale))
    sc_tm = _int4_scale_tm(scale, block_n=BLOCK_N)
    N, K = packed.shape[1], packed.shape[2] * 2
    dec = decode_int4_to_int8(packed, N, K).view(torch.uint8).contiguous()
    return dec, sc_tm

HIDDEN = 4096
IH = 2048
NUM_EXPERTS = 32
NUM_TOPK = 1
CLAMP = 10.0
WEIGHT_SCALE = 0.05
BLOCK_N = int(os.environ.get("DG_TEST_BLOCK_N", "128"))
BATCHES = [int(x) for x in sys.argv[1:]] or [1280, 2048, 4096]


def _worker(local_rank: int, num_local_ranks: int) -> None:
    from deep_gemm.utils.dist import init_dist
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(20260708)

    l1_bf = torch.randn((NUM_EXPERTS, 2 * IH, HIDDEN), dtype=torch.bfloat16,
                        device="cuda") * WEIGHT_SCALE
    l2_bf = torch.randn((NUM_EXPERTS, HIDDEN, IH), dtype=torch.bfloat16,
                        device="cuda") * WEIGHT_SCALE
    if QOQ:
        from deep_gemm.quantization_mxfp4 import mxfp4_fuse_packed_with_scale_tile_major
        l1_dec, l1_plane, l1_dequant, l1_w4 = quantize_qoq(l1_bf)
        l2_dec, l2_plane, l2_dequant, l2_w4 = quantize_qoq(l2_bf)
        if PRE:
            l1_dec, l1_plane = _il1((l1_dec, l1_plane))
            transformed_l1 = (l1_dec.view(torch.uint8).contiguous(),
                              _qoq_plane_tm(l1_plane, block_n=BLOCK_N))
            transformed_l2 = (l2_dec.view(torch.uint8).contiguous(),
                              _qoq_plane_tm(l2_plane, block_n=BLOCK_N))
        else:
            # Inline QoQ: 4-bit packed values + [s2|s1] coeff plane, fused
            # exactly like the fp32-scale path.
            l1_pk, l2_pk = _pack_w4(l1_w4), _pack_w4(l2_w4)
            l1_pk, l1_plane = _il1((l1_pk, l1_plane))
            def _fuse(pk, plane):
                return (mxfp4_fuse_packed_with_scale_tile_major(
                            pk.contiguous(), _qoq_plane_tm(plane, block_n=BLOCK_N),
                            block_k=128, use_prmt_groups=True, use_rf_fragments=True),
                        _qoq_plane_tm(plane, block_n=BLOCK_N))
            transformed_l1 = _fuse(l1_pk, l1_plane)
            transformed_l2 = _fuse(l2_pk, l2_plane)
        l1_packed = l2_packed = None
    else:
        l1_packed, l1_scale = quantize_to_int4(l1_bf)
        l1_dequant = dequant_int4(l1_packed, l1_scale, 2 * IH, HIDDEN)
        l2_packed, l2_scale = quantize_to_int4(l2_bf)
        l2_dequant = dequant_int4(l2_packed, l2_scale, HIDDEN, IH)
    del l1_bf, l2_bf
    if QOQ:
        print("QOQ mode: two-level scale, s2 folded", flush=True)
    elif PRE:
        transformed_l1 = transform_int4_pre(l1_packed, l1_scale, is_l1=True)
        transformed_l2 = transform_int4_pre(l2_packed, l2_scale, is_l1=False)
    else:
        transformed_l1 = transform_int4(l1_packed, l1_scale, is_l1=True, block_n=BLOCK_N)
        transformed_l2 = transform_int4(l2_packed, l2_scale, is_l1=False, block_n=BLOCK_N)
    if PRE:
        print("PRE mode: pre-decoded int8 weights", flush=True)

    max_m = max(BATCHES)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, NUM_EXPERTS, max_m, NUM_TOPK, HIDDEN, IH,
        mma_type="fp8xmxfp4", activation="swiglu")

    ok = True
    for m in BATCHES:
        x_bf = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
        x_i8, x_ts = per_token_int8(x_bf)
        x_sf = x_ts.unsqueeze(1).repeat(1, HIDDEN // 128).contiguous()
        buffer.x[:m].view(torch.uint8).copy_(x_i8.view(torch.uint8))
        buffer.x_sf[:m].copy_(x_sf)
        buffer.topk_idx[:m].copy_(
            torch.zeros((m, NUM_TOPK), dtype=torch.int32, device="cuda"))
        buffer.topk_weights[:m].copy_(
            torch.ones((m, NUM_TOPK), dtype=torch.float32, device="cuda"))

        cumulative_stats = torch.zeros(NUM_EXPERTS, dtype=torch.int, device="cuda")
        y_kernel = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
        deep_gemm.mxfp4_mega_moe(
            y_kernel, transformed_l1, transformed_l2, buffer,
            cumulative_local_expert_recv_stats=cumulative_stats,
            recipe=(128, BLOCK_N, 128), activation="swiglu", activation_clamp=CLAMP,
            fast_math=True)
        torch.cuda.synchronize()

        # Vectorized reference (single expert, weight 1.0)
        x_ref = x_i8.float() * x_ts.float().unsqueeze(1)
        l1_out = x_ref @ l1_dequant[0].float().t()          # (m, 2IH)
        gate = l1_out[:, :IH].clamp(max=CLAMP)
        up = l1_out[:, IH:].clamp(min=-CLAMP, max=CLAMP)
        inter = _silu(gate) * up                            # (m, IH)
        iv = inter.view(m, IH // 64, 64)
        isf = (iv.abs().amax(dim=-1) / 127.0).clamp_min(
            torch.finfo(torch.float32).tiny)                # (m, IH/64)
        iq = q8_ref(iv, isf)
        inter_qdq = (iq * isf.unsqueeze(-1)).view(m, IH)
        y_ref = (inter_qdq @ l2_dequant[0].float().t()).to(torch.bfloat16).float()

        cos = torch.nn.functional.cosine_similarity(
            y_kernel.float(), y_ref, dim=-1)
        nr = (y_kernel.float().norm() / y_ref.norm().clamp_min(1e-30)).item()
        finite = bool(torch.isfinite(y_kernel).all().item())
        line = (f"M={m}: cosine_min={cos.min().item():.6f} "
                f"cosine_mean={cos.mean().item():.6f} norm_ratio={nr:.6f} "
                f"finite={finite}")
        passed = finite and cos.min().item() > 0.999 and abs(nr - 1.0) < 0.01
        ok = ok and passed
        print(("PASS " if passed else "FAIL ") + line, flush=True)

    print(f"LARGE-M SMOKE {'PASS' if ok else 'FAIL'}", flush=True)
    dist.barrier(group=group)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(_worker, args=(1,), nprocs=1, join=True)
