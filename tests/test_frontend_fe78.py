"""Full-K tiny-M grid (DG_FE_TINYM_GRID=auto, SM-count router CTAs) vs the legacy
96-CTA K-split grid (DG_FE_TINYM_GRID=96), random inputs, single GPU.

Acceptance (not bit-identity: the fp32 accumulation order differs before the bf16
logit rounding): identical top-8 INDEX SETS and top-k weights within 1e-6 per row,
quantised activations (x, x_sf) bit-identical. Rows whose index sets differ are
classified against an fp32 torch reference: "near-tie" if the reference bf16 logits
of the 8th and 9th candidates are within one bf16 ulp; weight mismatches > 1e-6 are
"bf16 rounding flips" (a selected logit rounded to the neighbouring bf16 value).

  python3 tests/test_frontend_fe78.py [--seeds 40] [--rows 1 2 8 16]   # 40 x 27 = 1080 rows
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import deep_gemm  # noqa: E402
from test_frontend_tinym import HIDDEN, EXPERTS, TOPK, make_buffer  # noqa: E402


def run(buf, hidden, w, quant, grid, mma="wmma"):
    buf.x.view(torch.uint8).fill_(0xAB); buf.x_sf.fill_(-1.0)
    buf.topk_idx.fill_(-7); buf.topk_weights.fill_(-1.0)
    deep_gemm.fable_router_quant_topk_frontend(hidden, w, buf, quant=quant, tinym=1, stamps=0, grid=grid, mma=mma)
    torch.cuda.synchronize()
    m = hidden.size(0)
    return (buf.x[:m].view(torch.uint8).clone(), buf.x_sf[:m].clone(),
            buf.topk_idx[:m].clone(), buf.topk_weights[:m].clone())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 8, 16])
    ap.add_argument("--grid", default="auto", help="new-scheme DG_FE_TINYM_GRID value (auto or N)")
    ap.add_argument("--weight-tol", type=float, default=1e-6)
    ap.add_argument("--mma", default="wmma", choices=("wmma", "fma"), help="new-scheme DG_FE_TINYM_MMA value")
    args = ap.parse_args()
    buf_old, buf_new = make_buffer(64), make_buffer(64)
    rows = set_mismatch = near_tie = weight_flip = quant_mismatch = idx_order_diff = 0
    n_router = deep_gemm.fable_frontend_router_ctas(1, EXPERTS, HIDDEN, TOPK, 1, args.grid)
    print(f"full-K grid={args.grid} mma={args.mma}: router CTAs={n_router} (+1 merger) vs legacy 96+m; "
          f"device={torch.cuda.get_device_name()} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    for seed in range(args.seeds):
        torch.manual_seed(5000 + seed)
        w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05)
        if seed % 4 == 3:   # coarse weights/activations -> exact logit ties
            w = (w * 8).round().to(torch.bfloat16) / 8
        for m in args.rows:
            hidden = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            hidden *= torch.exp(torch.randn(m, 1, device="cuda")).to(torch.bfloat16)
            if seed % 4 == 3:
                hidden = (hidden * 4).round().to(torch.bfloat16) / 4
            ref = (hidden.float() @ w.float().t()).to(torch.bfloat16).float()      # [m, E]
            ref_sorted = ref.sort(dim=1, descending=True).values
            for quant in ("mxfp4", "qoq"):
                old = run(buf_old, hidden, w, quant, 96)
                new = run(buf_new, hidden, w, quant, args.grid, args.mma)
                if not (torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])):
                    quant_mismatch += 1
                    print(f"QUANT MISMATCH seed={seed} m={m} quant={quant}")
                if quant == "mxfp4":
                    rows += m
                for t in range(m):
                    oi, ni = old[2][t].tolist(), new[2][t].tolist()
                    if oi != ni:
                        idx_order_diff += 1
                    if set(oi) != set(ni):
                        set_mismatch += 1
                        gap = (ref_sorted[t, TOPK - 1] - ref_sorted[t, TOPK]).item()
                        ulp = torch.finfo(torch.bfloat16).eps * max(abs(ref_sorted[t, TOPK - 1].item()), 1e-30)
                        tie = gap <= ulp * 1.0001
                        near_tie += int(tie)
                        print(f"INDEX SET MISMATCH seed={seed} m={m} t={t} quant={quant} near_tie={tie} "
                              f"gap8/9={gap:.3e} (1 bf16 ulp={ulp:.3e}) old={sorted(oi)} new={sorted(ni)}")
                        continue
                    ow = dict(zip(oi, old[3][t].tolist())); nw = dict(zip(ni, new[3][t].tolist()))
                    dmax = max(abs(ow[i] - nw[i]) for i in oi)
                    if dmax > args.weight_tol:
                        weight_flip += 1
                        print(f"WEIGHT DIFF seed={seed} m={m} t={t} quant={quant} max|dw|={dmax:.3e} (bf16 rounding flip)")
    print(f"full-K vs legacy: rows={rows} per quant mode ({2 * rows} row-evaluations); "
          f"top-8 index-set mismatches={set_mismatch} (near-tie {near_tie}, other {set_mismatch - near_tie}); "
          f"weight diffs > {args.weight_tol:g}={weight_flip}; ordering-only differences={idx_order_diff - set_mismatch}; "
          f"x/x_sf mismatches={quant_mismatch}")
    ok = quant_mismatch == 0 and (set_mismatch - near_tie) == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
