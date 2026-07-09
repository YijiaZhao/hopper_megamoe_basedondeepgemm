"""Element-wise instrument for the W4A8-int L2 path (cosine 0.41 bug).

Single process, M=1, topk=1, forced route to local expert 0. Splits the
pipeline at the intermediate pool: compares the kernel-written int8
intermediate + per-64 SF against a Python reference, then the final y.

Run inside dg_dev with DG_W4A8_INT=1 DG_W4A8_INT_L2=1.
"""

import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_mxfp4 import mxfp4_fuse_packed_with_scale_tile_major
from deep_gemm.mega import _interleave_l1_weights as _il1
from deep_gemm.utils.dist import init_dist

INT4_GROUP = 128

HIDDEN = 4096
IH = 2048
NUM_EXPERTS = 32
NUM_TOPK = 1
M = 1
CLAMP = 10.0
WEIGHT_SCALE = 0.05


def quantize_to_int4(weight, group_size=INT4_GROUP):
    *outer, K = weight.shape
    G = K // group_size
    w = weight.float().view(*outer, G, group_size)
    scale = (w.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-30)
    q = (w / scale).round().clamp(-7, 7).to(torch.int8).view(*outer, K)
    scale = scale.squeeze(-1).to(torch.float32)
    nib = (q & 0x0F).to(torch.uint8).view(*outer, K // 8, 8)
    packed = (nib[..., 4:8] | (nib[..., 0:4] << 4)).view(*outer, K // 2).contiguous()
    return packed, scale


def dequant_int4(packed, scale, N, K, group_size=INT4_GROUP):
    E = packed.shape[0]
    b = packed.view(E, N, K // 2)
    def s4(x):
        x = x.to(torch.int16); return torch.where(x >= 8, x - 16, x)
    q = torch.empty(E, N, K, dtype=torch.int16, device=packed.device)
    qv = q.view(E, N, K // 8, 8)
    pv = b.view(E, N, K // 8, 4)
    qv[..., 0:4] = s4(pv >> 4); qv[..., 4:8] = s4(pv & 0x0F)
    return (q.float().view(E, N, K // group_size, group_size)
            * scale.view(E, N, K // group_size, 1)).view(E, N, K)


def _int4_scale_tm(scale_fp32, block_n=128):
    E, N, Kb = scale_fp32.shape
    b = scale_fp32.contiguous().view(torch.uint8).view(E, N, Kb, 4)
    return (b.view(E, N // block_n, block_n, Kb, 4).permute(0, 1, 3, 2, 4).contiguous())


def transform_int4(packed, scale, is_l1, block_n=128):
    if is_l1:
        packed, scale = _il1((packed, scale))
    sc_tm = _int4_scale_tm(scale, block_n=block_n)
    pk = mxfp4_fuse_packed_with_scale_tile_major(
        packed.contiguous(), sc_tm, block_k=128,
        use_prmt_groups=True, use_rf_fragments=True)
    return pk, sc_tm


def per_token_int8(x_bf):
    x = x_bf.float()
    s = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (x / s).round().clamp(-127, 127).to(torch.int8)
    return q, s.squeeze(-1)


def _silu(x):
    return x * torch.sigmoid(x)


def q8_ref(v, sf):
    # Mirrors the kernel's q8: round-to-nearest-even, clamp +-126, -1 -> 0.
    q = (v / sf.unsqueeze(-1)).round().clamp(-126, 126)
    return torch.where(q == -1, torch.zeros_like(q), q)


def _worker(local_rank: int, num_local_ranks: int) -> None:
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(20260708)

    l1_bf = torch.randn((NUM_EXPERTS, 2 * IH, HIDDEN), dtype=torch.bfloat16,
                        device="cuda") * WEIGHT_SCALE
    l1_packed, l1_scale = quantize_to_int4(l1_bf)
    l1_dequant = dequant_int4(l1_packed, l1_scale, 2 * IH, HIDDEN)
    l2_bf = torch.randn((NUM_EXPERTS, HIDDEN, IH), dtype=torch.bfloat16,
                        device="cuda") * WEIGHT_SCALE
    l2_packed, l2_scale = quantize_to_int4(l2_bf)
    l2_dequant = dequant_int4(l2_packed, l2_scale, HIDDEN, IH)
    transformed_l1 = transform_int4(l1_packed, l1_scale, is_l1=True)
    transformed_l2 = transform_int4(l2_packed, l2_scale, is_l1=False)

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, NUM_EXPERTS, 32, NUM_TOPK, HIDDEN, IH,
        mma_type="fp8xmxfp4", activation="swiglu")

    x_bf = torch.randn((M, HIDDEN), dtype=torch.bfloat16, device="cuda")
    x_i8, x_ts = per_token_int8(x_bf)
    x_sf = x_ts.unsqueeze(1).repeat(1, HIDDEN // 128).contiguous()
    topk_idx = torch.zeros((M, NUM_TOPK), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((M, NUM_TOPK), dtype=torch.float32, device="cuda")

    buffer.x[:M].view(torch.uint8).copy_(x_i8.view(torch.uint8))
    buffer.x_sf[:M].copy_(x_sf)
    buffer.topk_idx[:M].copy_(topk_idx)
    buffer.topk_weights[:M].copy_(topk_weights)

    cumulative_stats = torch.zeros(NUM_EXPERTS, dtype=torch.int, device="cuda")
    y_kernel = torch.zeros((M, HIDDEN), dtype=torch.bfloat16, device="cuda")
    deep_gemm.mxfp4_mega_moe(
        y_kernel, transformed_l1, transformed_l2, buffer,
        cumulative_local_expert_recv_stats=cumulative_stats,
        recipe=None, activation="swiglu", activation_clamp=CLAMP,
        fast_math=True)
    torch.cuda.synchronize()

    print(f"cum_stats nonzero: {cumulative_stats.nonzero().flatten().tolist()}, "
          f"counts: {cumulative_stats[cumulative_stats > 0].tolist()}", flush=True)
    print(f"l2_acts: shape={tuple(buffer.l2_acts.shape)} dtype={buffer.l2_acts.dtype}",
          flush=True)
    print(f"l2_acts_sf: shape={tuple(buffer.l2_acts_sf.shape)} dtype={buffer.l2_acts_sf.dtype}",
          flush=True)

    # ---- reference ----
    x_ref = x_i8.float() * x_ts.float().unsqueeze(1)
    l1_out = l1_dequant[0].float() @ x_ref[0]
    gate, up = l1_out[:IH], l1_out[IH:]
    gate = gate.clamp(max=CLAMP)
    up = up.clamp(min=-CLAMP, max=CLAMP)
    inter = _silu(gate) * up  # route weight = 1.0
    iv = inter.view(IH // 64, 64)
    isf = (iv.abs().amax(dim=-1) / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    iq = q8_ref(iv, isf)  # (IH/64, 64) int values
    inter_qdq = (iq * isf.unsqueeze(-1)).view(IH)
    y_ref = (l2_dequant[0].float() @ inter_qdq).to(torch.bfloat16).float()

    # ---- stage 1: intermediate int8 + SF vs pool ----
    acts = buffer.l2_acts.view(torch.uint8)
    acts2d = acts.view(-1, IH) if acts.numel() % IH == 0 else None
    if acts2d is not None:
        nz_rows = (acts2d != 0).any(dim=-1).nonzero().flatten()
        print(f"intermediate pool rows nonzero: {nz_rows.tolist()[:16]}", flush=True)
        if len(nz_rows) > 0:
            row = acts2d[nz_rows[0]].view(torch.int8).float()
            ref_row = iq.view(IH).float()
            match = (row == ref_row).float().mean().item()
            diff_idx = (row != ref_row).nonzero().flatten()
            print(f"int8 intermediate exact-match ratio: {match:.4f} "
                  f"({len(diff_idx)} mismatches)", flush=True)
            if len(diff_idx) > 0:
                head = diff_idx[:16].tolist()
                print(f"first mismatch cols: {head}", flush=True)
                for c in head[:8]:
                    print(f"  col {c}: kernel={row[c].item():.0f} ref={ref_row[c].item():.0f}",
                          flush=True)
                # error histogram per 64-col group
                per64 = (row != ref_row).view(IH // 64, 64).sum(dim=-1)
                print(f"mismatches per 64-col group: {per64.tolist()}", flush=True)

    # Physical order of the SF pool: the python view is (padded_tokens, k)
    # with strides (1, padded_tokens) -> .t() is contiguous == raw memory.
    sf_phys = buffer.l2_acts_sf.t().contiguous().view(-1)
    nz_phys = (sf_phys != 0).nonzero().flatten()
    print(f"sf pool phys nonzero={len(nz_phys)} first: {nz_phys[:8].tolist()}",
          flush=True)
    stride = int(nz_phys[1].item() - nz_phys[0].item()) if len(nz_phys) > 1 else 0
    print(f"inferred kSFPoolStrideTokens={stride}", flush=True)
    ker_sf = sf_phys[nz_phys[:IH // 64]]
    print(f"kernel SF[:8]:  {[f'{v:.5e}' for v in ker_sf[:8].tolist()]}", flush=True)
    print(f"ref amax/127[:8]: {[f'{v:.5e}' for v in isf[:8].tolist()]}", flush=True)
    rel = ((ker_sf - isf) / isf).abs()
    print(f"SF vs amax/127: max_rel={rel.max().item():.3e} "
          f"mean_rel={rel.mean().item():.3e}", flush=True)
    # e4m3-style SF would be amax/448-ish; check kernel_sf * 448 / amax ~ 1
    rel448 = ((ker_sf * 448.0 / 127.0 - isf) / isf).abs()
    print(f"SF vs amax/448: max_rel={rel448.max().item():.3e} "
          f"mean_rel={rel448.mean().item():.3e}", flush=True)

    # Decisive: which byte interpretation reconstructs the reference
    # intermediate? (a) int8 * ker_sf, (b) fp8 e4m3 * ker_sf.
    if acts2d is not None and len(nz_rows) > 0:
        raw = acts2d[nz_rows[0]]
        as_int = raw.view(torch.int8).float().view(IH // 64, 64)
        as_fp8 = raw.view(torch.float8_e4m3fn).float().view(IH // 64, 64)
        rec_int = (as_int * ker_sf.unsqueeze(-1)).view(IH)
        rec_fp8 = (as_fp8 * ker_sf.unsqueeze(-1)).view(IH)
        cos_int = torch.nn.functional.cosine_similarity(
            rec_int, inter.view(IH), dim=-1).item()
        cos_fp8 = torch.nn.functional.cosine_similarity(
            rec_fp8, inter.view(IH), dim=-1).item()
        print(f"reconstruct cosine vs ref float intermediate: "
              f"int8*sf={cos_int:.4f}  fp8*sf={cos_fp8:.4f}", flush=True)
        torch.save({
            "raw_bytes": raw.view(torch.uint8).cpu(),
            "ker_sf": ker_sf.cpu(), "ref_isf": isf.cpu(),
            "ref_iq": iq.cpu(), "ref_inter": inter.cpu(),
            "y_kernel": y_kernel.cpu(), "y_ref": y_ref.cpu(),
            "x_i8": x_i8.cpu(), "x_ts": x_ts.cpu(),
        }, "/tmp/int_l2_dump.pt")
        print("saved /tmp/int_l2_dump.pt", flush=True)

    # ---- stage 2: final y ----
    cos = torch.nn.functional.cosine_similarity(
        y_kernel[0].float(), y_ref, dim=-1).item()
    nr = (y_kernel[0].float().norm() / y_ref.norm().clamp_min(1e-30)).item()
    print(f"y: cosine={cos:.4f} norm_ratio={nr:.4f}", flush=True)
    err = (y_kernel[0].float() - y_ref).abs()
    per128 = err.view(HIDDEN // 128, 128).mean(dim=-1)
    ref128 = y_ref.abs().view(HIDDEN // 128, 128).mean(dim=-1).clamp_min(1e-30)
    print("per-128-out-col mean_abs_err / mean_abs_ref:", flush=True)
    print([f"{(e / r).item():.2f}" for e, r in zip(per128, ref128)], flush=True)

    # per-k-block contribution check: recompute y using kernel's own
    # intermediate (if it matched, identical to ref; else isolate)
    dist.barrier(group=group)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(_worker, args=(1,), nprocs=1, join=True)
