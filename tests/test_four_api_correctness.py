"""Exact-quantized-reference validation for the four H20 MegaMoE APIs.

The test intentionally calls MXFP4 and QoQ in the same process, so it also
checks that the explicit APIs do not depend on process-global DG_W4A8_INT.

--frontend none (default): balanced synthetic routing + per-token quantisation.
--frontend fe: the Fable frontend (router + top-k + softmax + quant, tiny-M path) produces
  x / x_sf / topk_idx / topk_weights from a bf16 hidden and a shared router weight; the
  reference uses those outputs (real, unbalanced routing). Fused APIs only.
--zero-rows K [--zero-ranks r,r,...]: the first K hidden rows of every listed rank (default: all
  ranks) are all-zero (the E2E harness's padding rows). With --frontend fe the pipeline runs twice
  per seed, FE knob zero_row_unrouted 0 then 1 (DG_FE_ZERO_ROW_UNROUTED; effective on the cc path,
  <= 2 rows per rank): the kernel output of every zero row must be exactly 0 with both, the FE must
  leave zero rows unrouted with the knob on (cc path), and x / x_sf / the routing / y of the other
  rows must be bit-identical between the two runs (ZERO_ROWS line; assertion).
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deep_gemm
from deep_gemm.mega import FE_PATH_CC
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


def _topk_key_bits(value: float, expert: int) -> int:
    """fable cc router key of a bf16 logit (orderable bf16 << 16 | 0xFFFF - expert), as a u32
    (same as tests/profile_four_api_h20.py topk_key_bits)."""
    b = torch.tensor(value, dtype=torch.bfloat16).view(torch.int16).item() & 0xFFFF
    o = (~b & 0xFFFF) if (b & 0x8000) else (b | 0x8000)
    return (o << 16) | (0xFFFF - expert)


def _topk_from_keys(keys_i32, m, e, topk):
    """Python decode of the cc router's compact key array: top-8 by key (value desc, smaller expert first on equal
    logits), fp32 softmax over the 8 selected bf16 logits -- what the Mega prologue selects under DG_FE_SELECT_IN_MEGA=1."""
    k = keys_i32.view(m, e).to(torch.int64) & 0xFFFFFFFF
    top = torch.topk(k, topk, dim=1).values
    expert = 0xFFFF - (top & 0xFFFF)
    o = top >> 16
    bits = torch.where((o & 0x8000) != 0, o & 0x7FFF, (~o) & 0xFFFF)
    logits = bits.to(torch.int16).view(torch.bfloat16).float()
    weights = torch.softmax(logits, dim=1)
    # all-zero keys = unrouted token (the Mega prologue writes topk_idx -1 / weight 0 for it)
    unrouted = (k[:, :1] == 0)
    expert = torch.where(unrouted, torch.full_like(expert, -1), expert)
    weights = torch.where(unrouted, torch.zeros_like(weights), weights)
    return expert.contiguous(), weights.contiguous()


def _balanced_routing(rank, m, local_experts, topk, experts, active=True):
    """Forced-balanced routing (DG_FE_FORCE_BALANCED=1), the assignment of the 2026-09-04 delivery's Mega-only scope
    and of the profile's DG_PROFILE_FORCE_BALANCED memcpy (tests/profile_four_api_h20.py forced_balanced_routing):
    token t of this rank (global token g = rank * m + t) sends slot s to expert s * local_experts + (g + 7 s) % local_experts,
    i.e. exactly one route to every EP rank, weight 1 / topk each. Returns (topk_idx [m, topk] int64,
    topk_weights [m, topk] fp32, keys [m * experts] int32: chosen experts logit 1.0, the others 0.0, so the Mega
    prologue selects them in slot order with softmax weight exactly 1 / topk). Inactive rank: -1 / 0 / all-zero keys."""
    idx = torch.full((m, topk), -1, dtype=torch.int64)
    wts = torch.zeros(m, topk, dtype=torch.float32)
    keys = torch.zeros(m, experts, dtype=torch.int64)
    if active:
        g = rank * m + torch.arange(m, dtype=torch.int64)[:, None]
        slot = torch.arange(topk, dtype=torch.int64)[None, :]
        idx = slot * local_experts + (g + slot * 7) % local_experts
        wts.fill_(1.0 / topk)
        keys[:] = torch.tensor([_topk_key_bits(0.0, x) for x in range(experts)], dtype=torch.int64)
        keys.scatter_(1, idx, torch.tensor([_topk_key_bits(1.0, x) for x in range(experts)], dtype=torch.int64)[idx])
    keys = keys & 0xFFFFFFFF
    keys = torch.where(keys >= 2 ** 31, keys - 2 ** 32, keys).to(torch.int32).reshape(-1)
    return idx.cuda(), wts.cuda(), keys.cuda()


def _run(api, args, rank, group):
    world = dist.get_world_size(group)
    local_experts = args.experts // world
    is_qoq = api.startswith("qoq")
    is_fused = api.endswith("fused")
    use_fe = args.frontend != "none"
    assert not use_fe or is_fused, "--frontend needs the fused APIs"
    # DG_FE_SELECT_IN_MEGA=1 (the E2E pipeline configuration): the FE stops after the 384 compact keys per token and
    # the fused Mega prologue selects the top-8 (cc router, <= 2 rows per rank only; more rows = full FE as usual).
    sel_in_mega = use_fe and os.environ.get("DG_FE_SELECT_IN_MEGA", "0") == "1" and args.tokens <= 2 and args.experts == 384
    # --zero-rows: rows of THIS rank that are all-zero; the FE leaves them unrouted on the cc path when the knob is on
    zero_ranks = list(range(world)) if args.zero_ranks == "all" else [int(r) for r in args.zero_ranks.split(",")]
    zero_mask = torch.zeros(args.tokens, dtype=torch.bool, device="cuda")
    if args.zero_rows > 0 and rank in zero_ranks:
        assert args.zero_rows <= args.tokens
        zero_mask[:args.zero_rows] = True
    fe_path_cc = use_fe and deep_gemm.fable_frontend_path(args.tokens, args.hidden, args.experts, args.topk) == FE_PATH_CC
    zr_knob = int(os.environ.get("DG_FE_ZERO_ROW_UNROUTED", "1")) if args.zero_row_unrouted is None else args.zero_row_unrouted
    fe_unroutes_zero = bool(zr_knob) and fe_path_cc
    zero_ab = use_fe and args.zero_rows > 0            # knob 0 then knob 1 in one process, per seed

    torch.manual_seed(20260805)
    router_weight = (torch.randn(args.experts, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
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
    # --hot-rows N: the first N global tokens (all on rank 0 when N <= tokens) send every
    # route to local expert 0, so local expert 0 of every rank receives N rows from ONE
    # source rank (N > 8 = a second BM8 pool block at tiny M).
    if args.hot_rows > 0:
        offsets = torch.where(token_ids[:, None] < args.hot_rows, torch.zeros_like(offsets), offsets)
    topk_idx = slots[None, :] * local_experts + offsets
    topk_weights = torch.full(
        (args.tokens, args.topk), 1.0 / args.topk,
        device="cuda", dtype=torch.float32)
    # --global-tokens M < world (tokens must be 1): mirror the profile scripts' owner
    # layout (tests/profile_four_api_h20.py local_tokens: one token on the first M // 2
    # ranks of each TP4 half, M=2 -> ranks 0,4; M=4 -> ranks 0,1,4,5); the other ranks
    # carry one padded row (topk_idx -1, weight 0). This is the only way to reach the
    # launches where the fused kernel's tiny-M schedulers (stream-K) actually activate.
    active = True
    if args.global_tokens is not None and args.global_tokens < world:
        assert args.tokens == 1, "--global-tokens < world needs --tokens 1"
        active = (rank % 4) < args.global_tokens // 2
        if not active:
            topk_idx.fill_(-1)
            topk_weights.zero_()

    if is_qoq:
        x_quant = lambda x: per_token_cast_to_int8(x, gran_k=128)          # noqa: E731
        if is_fused:
            q1, q2 = quantize_to_qoq(w1_bf), quantize_to_qoq(w2_bf)
            d1, d2 = dequantize_qoq_to_fp32(*q1), dequantize_qoq_to_fp32(*q2)
            weights = deep_gemm.transform_qoq_weights_for_mega_moe_fused(q1, q2)
        else:
            q1, q2 = quantize_to_qoq_int4(w1_bf), quantize_to_qoq_int4(w2_bf)
            d1, d2 = dequantize_qoq_int4(*q1), dequantize_qoq_int4(*q2)
            weights = deep_gemm.transform_qoq_int4_weights_for_mega_moe_sm90(q1, q2, block_n=128)
    else:
        x_quant = lambda x: per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)   # noqa: E731
        if is_fused:
            q1, q2 = quantize_to_mxfp4_fused(w1_bf), quantize_to_mxfp4_fused(w2_bf)
            d1 = dequantize_mxfp4_kernel_exact_fp32(q1[0], q1[1], mxfp4_row_reference_exponent(q1[1]))
            d2 = dequantize_mxfp4_kernel_exact_fp32(q2[0], q2[1], mxfp4_row_reference_exponent(q2[1]))
            weights = deep_gemm.transform_mxfp4_weights_for_mega_moe_fused(q1, q2)
        else:
            q1, q2 = quantize_to_mxfp4_split(w1_bf), quantize_to_mxfp4_split(w2_bf)
            d1, d2 = dequantize_mxfp4_split(*q1), dequantize_mxfp4_split(*q2)
            weights = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(q1, q2, block_n=128)

    def x_reference(xq, xs):
        if is_qoq:
            return int8_bytes_to_float(xq) * xs[:, :1]
        return (xq.float().view(args.tokens, args.hidden // 128, 128)
                * xs.float().unsqueeze(-1)).view(args.tokens, args.hidden)

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
        m = args.tokens
        syn_idx, syn_w = topk_idx, topk_weights
        failures = []
        agg = {"cos_min": 1.0, "max_abs": 0.0, "norm_lo": 9.0, "norm_hi": 0.0, "agree": 0, "tokens": 0}
        for seed in range(args.seed, args.seed + args.seeds):
            topk_idx, topk_weights = syn_idx, syn_w
            torch.manual_seed(seed + rank * 1000003)
            x_bf = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
            x_bf[zero_mask] = 0
            xq, xs = x_quant(x_bf)
            fe_idx = fe_w = None            # the FE's own top-8 (before any override), for the ROUTER_REF report
            y = torch.zeros(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
            if zero_ab:
                # knob OFF pipeline first (FE routes the zero rows onto experts 0..7 with weight 1/8): FE outputs + y
                deep_gemm.fable_router_quant_topk_frontend(
                    x_bf, router_weight, buffer, quant="qoq" if is_qoq else "mxfp4", select_in_mega=int(sel_in_mega),
                    zero_row_unrouted=0)
                torch.cuda.synchronize()
                off = {"x": buffer.x[:m].clone(), "x_sf": buffer.x_sf[:m].clone()}
                if sel_in_mega:
                    off["idx"], off["w"] = _topk_from_keys(deep_gemm.fable_frontend_keys(buffer)[:m * args.experts], m, args.experts, args.topk)
                else:
                    off["idx"], off["w"] = buffer.topk_idx[:m].clone(), buffer.topk_weights[:m].clone()
                kernel(y, *weights, buffer, activation_clamp=args.activation_clamp)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                off["y"] = y.clone()
                y.zero_()
            if use_fe:
                # Standalone Fable frontend -> buffer views; these define the reference routing unless overridden.
                deep_gemm.fable_router_quant_topk_frontend(
                    x_bf, router_weight, buffer, quant="qoq" if is_qoq else "mxfp4", select_in_mega=int(sel_in_mega),
                    zero_row_unrouted=zr_knob)
                torch.cuda.synchronize()
                xq, xs = buffer.x[:m].clone(), buffer.x_sf[:m].clone()
                if sel_in_mega:
                    keys = deep_gemm.fable_frontend_keys(buffer)
                    fe_idx, fe_w = _topk_from_keys(keys[:m * args.experts], m, args.experts, args.topk)
                else:
                    fe_idx, fe_w = buffer.topk_idx[:m].clone(), buffer.topk_weights[:m].clone()
                if args.force_balanced:
                    # DG_FE_FORCE_BALANCED=1: the FE ran; its routing is overridden (same buffers the profile's
                    # DG_PROFILE_FORCE_BALANCED memcpy overwrites) with the balanced assignment of _balanced_routing.
                    b_idx, b_w, b_keys = _balanced_routing(rank, m, local_experts, args.topk, args.experts, active)
                    if sel_in_mega:
                        keys[:m * args.experts].copy_(b_keys)
                    else:
                        buffer.topk_idx[:m].copy_(b_idx)
                        buffer.topk_weights[:m].copy_(b_w)
                elif not active:
                    # owner layout, inactive rank: the FE routed the row; unroute it (topk_idx -1 / all-zero keys)
                    if sel_in_mega:
                        keys[:m * args.experts].zero_()
                    else:
                        buffer.topk_idx[:m].fill_(-1)
                        buffer.topk_weights[:m].zero_()
                torch.cuda.synchronize()
                if not sel_in_mega:
                    topk_idx, topk_weights = buffer.topk_idx[:m].clone(), buffer.topk_weights[:m].clone()
            else:
                buffer.x[:m].copy_(xq)
                buffer.x_sf[:m].copy_(xs)
                buffer.topk_idx[:m].copy_(topk_idx)
                buffer.topk_weights[:m].copy_(topk_weights)
            if is_fused:
                kernel(y, *weights, buffer, activation_clamp=args.activation_clamp)
            else:
                kernel(
                    y, *weights, buffer, recipe=(128, 128, 128),
                    activation_clamp=args.activation_clamp)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            if sel_in_mega:
                # the Mega prologue wrote the selected top-8 into the buffer: this IS the routing the kernel used
                topk_idx, topk_weights = buffer.topk_idx[:m].clone(), buffer.topk_weights[:m].clone()
                if not args.force_balanced:
                    # (collective on every rank; an inactive rank contributes 0)
                    n_bad = int((torch.sort(topk_idx, 1).values != torch.sort(fe_idx, 1).values).any(1).sum()) if active else 0
                    n_bad = torch.tensor([n_bad], device="cuda"); dist.all_reduce(n_bad, group=group)
                    if rank == 0:
                        print(f"SELECT_IN_MEGA api={api} tokens={m}: Mega-prologue top-8 vs python decode of the FE key array: "
                              f"{int(n_bad)} tokens differ", flush=True)
            if rank == 0:
                print(f"ROUTING api={api} tokens/rank={m} global_tokens={args.global_tokens or m * world} frontend={args.frontend} "
                      f"select_in_mega={int(sel_in_mega)} force_balanced={int(args.force_balanced)} reference={args.reference} "
                      f"zero_rows={args.zero_rows} zero_row_unrouted={zr_knob} (effective={int(fe_unroutes_zero)})", flush=True)
            if zero_ab:
                # Zero-row gate (collective). Zero rows: y == 0 exactly with both knob settings; knob on (cc path): the FE
                # left them unrouted. Other rows: x / x_sf / routing / y bit-identical between the knob-0 and knob-1 runs.
                nz = ~zero_mask
                y_zero_bad = int((y[zero_mask].view(torch.int16) != 0).sum() + (off["y"][zero_mask].view(torch.int16) != 0).sum())
                fe_unrouted = int((fe_idx[zero_mask] < 0).all(dim=1).sum()) if zero_mask.any() else 0
                fe_unrouted_bad = int(zero_mask.sum()) - fe_unrouted if fe_unroutes_zero else 0
                fe_routed_bad = fe_unrouted if not fe_unroutes_zero else 0     # knob off / non-cc path: zero rows stay routed
                x_diff = int((xq.view(torch.uint8) != off["x"].view(torch.uint8)).sum() + (xs != off["x_sf"]).sum())
                route_diff = int((fe_idx[nz] != off["idx"][nz]).sum() + (fe_w[nz] != off["w"][nz]).sum())
                y_diff_elems = int((y[nz].view(torch.int16) != off["y"][nz].view(torch.int16)).sum())
                y_diff_max = (y[nz].float() - off["y"][nz].float()).abs().max() if nz.any() else torch.zeros((), device="cuda")
                st = torch.tensor([int(zero_mask.sum()), fe_unrouted, y_zero_bad, fe_unrouted_bad, fe_routed_bad, x_diff, route_diff,
                                   y_diff_elems, int(nz.sum()) * args.hidden], device="cuda", dtype=torch.int64)
                dist.all_reduce(st, group=group)
                dist.all_reduce(y_diff_max, op=dist.ReduceOp.MAX, group=group)
                st = st.tolist()
                ok_zero = st[2] == 0 and st[3] == 0 and st[4] == 0 and st[5] == 0 and st[6] == 0 and st[7] == 0
                if rank == 0:
                    print(f"ZERO_ROWS api={api} seed={seed} zero_rows={st[0]} fe_unrouted={st[1]} y_zero_nonzero_elems={st[2]} "
                          f"fe_unrouted_missing={st[3]} fe_unexpected_unrouted={st[4]} | knob0 vs knob1 on the other rows: "
                          f"x/x_sf diffs={st[5]} routing diffs={st[6]} y diff elems={st[7]}/{st[8]} max|dy|={y_diff_max.item():.3g} "
                          f"-> {'PASS' if ok_zero else 'FAIL'}", flush=True)
                if not ok_zero:
                    failures.append((seed, {"zero_rows": st}))
            if use_fe and args.router_ref == "torch":
                # Pure-torch router reference, independent of the FE kernel: bf16 router GEMM (fp32 accumulate) ->
                # bf16-rounded logits -> top-8 (value desc, index asc on ties) -> fp32 softmax over the 8 selected
                # bf16 logits (the FE's normalisation); activations quantised by the torch per-token casts. The FE's own
                # outputs are compared to it (per-token top-8 index-set agreement, per-token weight max |diff|, x / x_sf
                # byte diffs) and -- unless the routing is overridden (force-balanced / inactive rank) -- the MoE
                # reference below routes with the TORCH top-8 / weights and dequantises the TORCH x, so y is checked
                # against a reference that contains none of our FE or Mega kernels.
                logits_ref = (x_bf.float() @ router_weight.float().T).to(torch.bfloat16).float()
                order = torch.sort(logits_ref, dim=-1, descending=True, stable=True).indices
                ref_idx = order[:, :args.topk].contiguous()
                ref_w = torch.softmax(torch.gather(logits_ref, 1, ref_idx), dim=-1).contiguous()
                if fe_unroutes_zero:
                    # the reference follows the FE contract: an all-zero row is unrouted (y = 0 either way, exact)
                    ref_idx[zero_mask] = -1
                    ref_w[zero_mask] = 0.0
                if is_qoq:
                    xq_ref, xs_ref = per_token_cast_to_int8(x_bf, gran_k=128)
                else:
                    xq_ref, xs_ref = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)
                fe_sorted, ref_sorted = torch.sort(fe_idx, dim=1), torch.sort(ref_idx, dim=1)
                agree = (fe_sorted.values == ref_sorted.values).all(dim=1)
                # per-token weight max |diff| over the union of both expert sets (a missing expert counts as weight 0)
                dense_fe = torch.zeros(m, args.experts, device="cuda").scatter_(1, fe_idx.clamp_min(0), fe_w)
                dense_ref = torch.zeros(m, args.experts, device="cuda").scatter_(1, ref_idx.clamp_min(0), ref_w)
                w_diff_tok = (dense_fe - dense_ref).abs().amax(dim=1)
                w_diff_agree = torch.where(agree, w_diff_tok, torch.zeros_like(w_diff_tok)).max()
                w_diff_all = w_diff_tok.max()
                x_row_diff = (xq.view(torch.uint8) != xq_ref.view(torch.uint8)).sum(dim=1)
                q_fe = xq.view(torch.int8).int() if is_qoq else xq.float()
                q_ref = xq_ref.view(torch.int8).int() if is_qoq else xq_ref.float()
                x_max_q_diff = (q_fe - q_ref).abs().float().max() if is_qoq else torch.zeros((), device="cuda")
                stats = torch.tensor([float(agree.sum()), float(m), float(x_row_diff.sum()),
                                      float((xs.float() != xs_ref.float()).sum()), float((x_row_diff > 0).sum())], device="cuda")
                dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
                for t_ in (w_diff_agree, w_diff_all, x_max_q_diff):
                    dist.all_reduce(t_, op=dist.ReduceOp.MAX, group=group)
                examples = []
                for t in (~agree).nonzero()[:, 0].tolist()[:2]:
                    only_fe = sorted(set(fe_idx[t].tolist()) - set(ref_idx[t].tolist()))
                    only_ref = sorted(set(ref_idx[t].tolist()) - set(fe_idx[t].tolist()))
                    examples.append(f"rank{rank} t{t}: FE-only experts {only_fe} (torch logit {[round(logits_ref[t, e_].item(), 5) for e_ in only_fe]}) "
                                    f"torch-only {only_ref} (logit {[round(logits_ref[t, e_].item(), 5) for e_ in only_ref]}) w_max_diff={w_diff_tok[t].item():.3g}")
                gathered = [None] * world
                dist.all_gather_object(gathered, examples, group=group)
                agg["agree"] += int(stats[0]); agg["tokens"] += int(stats[1])
                if rank == 0:
                    print(f"ROUTER_REF api={api} seed={seed} tokens={m} top8_set_agree={int(stats[0])}/{int(stats[1])} "
                          f"weight_max_abs_diff(agreeing)={w_diff_agree.item():.3g} weight_max_abs_diff(all)={w_diff_all.item():.3g} "
                          f"x_bytes_diff={int(stats[2])} x_rows_with_diff={int(stats[4])} x_max_|dq|={x_max_q_diff.item():.0f} x_sf_diff={int(stats[3])} "
                          f"(FE vs torch bf16-GEMM/top-8/softmax/per-token cast reference)", flush=True)
                    for ex in sum(gathered, [])[:4]:
                        print(f"  ROUTER_REF_DISAGREE {ex}", flush=True)
                xq, xs = xq_ref, xs_ref
                if not args.force_balanced and active:
                    topk_idx, topk_weights = ref_idx, ref_w
            x_ref_local = x_reference(xq, xs)

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
                    if expert < 0 or expert // local_experts != rank:
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
            if (use_fe or args.slot_check) and is_fused:
                # Per-(token, slot) kernel partials (the scattered L2 outputs the combine sums) vs the
                # reference route: pinpoints a wrong slot (expert / destination rank) on this rank.
                part = buffer.combine_partials[:, :args.tokens].float()          # [slot, token, hidden]
                bad = []
                for t in range(args.tokens):
                    for slot in range(args.topk):
                        e_idx = int(topk_idx[t, slot])
                        r = routes[rank * args.tokens + t, slot]
                        k = part[slot, t]
                        c = torch.nn.functional.cosine_similarity(k, r, dim=0).item() if r.abs().max() > 0 else 1.0
                        if c < 0.999:
                            # rows the destination expert received (global token ids), incl. this one
                            sharers = (idx_all == e_idx).nonzero()[:, 0].tolist()
                            # Best match among the routes of the tokens sharing this expert (a
                            # permuted token mapping shows as cos ~ 1 against another row).
                            best = ""
                            if args.slot_check:
                                hits = (idx_all == e_idx).nonzero().tolist()
                                cands = [(torch.nn.functional.cosine_similarity(k, routes[tt, ss], dim=0).item(), tt)
                                         for tt, ss in hits if routes[tt, ss].abs().max() > 0]
                                if cands:
                                    bc, bt = max(cands)
                                    best = f":best=t{bt}({bc:.4f})"
                            bad.append(f"t{t}s{slot}:e{e_idx}@r{e_idx // local_experts}:cos={c:.4f}:w={float(topk_weights[t, slot]):.4f}"
                                       f":|ref|={r.abs().max().item():.3g}:|ker|={k.abs().max().item():.3g}{best}:rows={sharers}")
                n_active = int(torch.unique(idx_all[(idx_all // local_experts) == rank]).numel())
                multi = int((torch.bincount(idx_all.flatten().clamp_min(0), minlength=args.experts)[rank * local_experts:(rank + 1) * local_experts] > 1).sum())
                bad.insert(0, f"[local active experts={n_active}, with >1 rows={multi}]")
                reports = [None] * world
                dist.all_gather_object(reports, f"rank{rank}: " + (" ".join(bad) if bad else "all slots cos>=0.999"), group=group)
                if rank == 0:
                    print("SLOT_CHECK api=" + api + "\n  " + "\n  ".join(reports), flush=True)
            if not active:
                # Padded row: the kernel output is not part of the check (finite only);
                # contribute a neutral (identical) pair to the collective metrics.
                assert torch.isfinite(y).all(), api
                y = torch.ones_like(y)
                ref = torch.ones_like(ref)
            if zero_mask.any():
                # all-zero rows: kernel and reference are exactly 0 (checked above / here); the cosine of two zero
                # vectors is undefined -> contribute a neutral pair to the collective metrics
                assert bool((ref[zero_mask] == 0).all()), api
                y = y.clone(); ref = ref.clone()
                y[zero_mask] = 1.0; ref[zero_mask] = 1.0
            metrics = _metrics(y, ref, group)
            if rank == 0:
                print(
                    f"RESULT api={api} seed={seed} finite={int(metrics['finite'])} "
                    f"max_abs={metrics['max_abs']:.8g} mean_abs={metrics['mean_abs']:.8g} "
                    f"cos_min={metrics['cos_min']:.10f} cos_mean={metrics['cos_mean']:.10f} "
                    f"norm_ratio={metrics['norm_ratio']:.10f}", flush=True)
            agg["cos_min"] = min(agg["cos_min"], metrics["cos_min"]); agg["max_abs"] = max(agg["max_abs"], metrics["max_abs"])
            agg["norm_lo"] = min(agg["norm_lo"], metrics["norm_ratio"]); agg["norm_hi"] = max(agg["norm_hi"], metrics["norm_ratio"])
            ok = metrics["finite"] and metrics["cos_min"] >= args.cosine_min and args.norm_ratio_min <= metrics["norm_ratio"] <= args.norm_ratio_max
            if not ok:
                failures.append((seed, metrics))
                if rank == 0:
                    print(f"SEED_FAIL api={api} seed={seed} {metrics}", flush=True)
        if rank == 0 and args.seeds > 1:
            print(f"SUMMARY api={api} seeds={args.seeds} tokens={m} cos_min={agg['cos_min']:.10f} max_abs={agg['max_abs']:.8g} "
                  f"norm_ratio=[{agg['norm_lo']:.6f}, {agg['norm_hi']:.6f}] top8_set_agree={agg['agree']}/{agg['tokens']} failing_seeds={len(failures)}", flush=True)
        assert not failures, (api, failures[:3])
    finally:
        buffer.destroy()
        dist.barrier(group=group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apis", nargs="+", default=[
        "mxfp4_mega_moe_split", "qoq_mega_moe_split",
        "mxfp4_mega_moe_fused", "qoq_mega_moe_fused"])
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--global-tokens", type=int, default=None,
                        help="< 8: owner-rank layout with --tokens 1 (2 -> ranks 0,4; 4 -> 0,1,4,5)")
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--seeds", type=int, default=1, help="run seeds seed .. seed+N-1 in one process (same weights, new hidden rows)")
    parser.add_argument("--cosine-min", type=float, default=0.99)
    parser.add_argument("--norm-ratio-min", type=float, default=0.97)
    parser.add_argument("--norm-ratio-max", type=float, default=1.03)
    parser.add_argument("--hot-rows", type=int, default=0,
                        help="route every slot of the first N global tokens to local expert 0 (N > 8: a second BM8 pool block)")
    parser.add_argument("--slot-check", action="store_true",
                        help="fused APIs: per-(token, slot) partial check (SLOT_CHECK) also without --frontend, with best-match row")
    parser.add_argument("--router-ref", choices=("fe", "torch"), default="fe",
                        help="--frontend fe: reference routing / activation quantisation from the FE outputs (fe) or from a pure-torch router + per-token cast (torch)")
    parser.add_argument("--reference", choices=("kernel", "torch-moe"), default="kernel",
                        help="torch-moe: pure-torch MoE reference from the same weights with the torch router's REAL top-8 routing "
                             "(= --frontend fe --router-ref torch; implies --frontend fe when --frontend is none)")
    parser.add_argument("--force-balanced", action="store_true", default=os.environ.get("DG_FE_FORCE_BALANCED", os.environ.get("DG_PROFILE_FORCE_BALANCED", "0")) == "1",
                        help="(env DG_FE_FORCE_BALANCED=1) --frontend fe: run the FE, then override its routing with the balanced "
                             "assignment (one route per EP rank, weight 1/8) in the buffers the Mega reads; the reference uses the same routing")
    parser.add_argument("--frontend", choices=("none", "fe"), default="none",
                        help="fused APIs: route/quantise with the Fable frontend (fe)")
    parser.add_argument("--zero-rows", type=int, default=0,
                        help="all-zero hidden rows per rank (the first K rows of every --zero-ranks rank); with --frontend fe the "
                             "ZERO_ROWS gate runs the pipeline with FE knob zero_row_unrouted 0 and 1 per seed")
    parser.add_argument("--zero-ranks", default="all", help="comma-separated ranks that carry the --zero-rows rows (default all)")
    parser.add_argument("--zero-row-unrouted", type=int, default=None,
                        help="FE knob for the checked run (default: env DG_FE_ZERO_ROW_UNROUTED, 1)")
    args = parser.parse_args()
    if args.reference == "torch-moe":
        args.router_ref = "torch"
        if args.frontend == "none":
            args.frontend = "fe"
    assert not args.force_balanced or args.frontend != "none", "--force-balanced overrides the FE routing: needs --frontend fe|fused"

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
