"""Bit-identity check: DG_FE_TINYM=1 (single-wave 3-stage config) vs the legacy
8-stage Fable frontend, on random inputs. Single GPU, no torch.distributed.

  python3 tests/test_frontend_tinym.py [--seeds 100] [--rows 1 2 4 8 16]

Compares x (fp8/int8 bytes), x_sf, topk_idx and topk_weights bit-for-bit for
both quant modes. Half of the seeds use coarsely-rounded weights to provoke
logit ties (tie-break rule: largest value, smallest expert id).
"""
import argparse
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import deep_gemm  # noqa: E402

HIDDEN, EXPERTS, TOPK = 3072, 384, 8


def make_buffer(max_m):
    return types.SimpleNamespace(
        x=torch.empty(max_m, HIDDEN, device="cuda", dtype=torch.float8_e4m3fn),
        x_sf=torch.empty(max_m, HIDDEN // 128, device="cuda", dtype=torch.float32),
        topk_idx=torch.empty(max_m, TOPK, device="cuda", dtype=torch.int64),
        topk_weights=torch.empty(max_m, TOPK, device="cuda", dtype=torch.float32),
    )


def run(buf, hidden, w, quant, tinym):
    buf.x.view(torch.uint8).fill_(0xAB); buf.x_sf.fill_(-1.0)
    buf.topk_idx.fill_(-7); buf.topk_weights.fill_(-1.0)
    deep_gemm.fable_router_quant_topk_frontend(hidden, w, buf, quant=quant, tinym=tinym, stamps=0)
    torch.cuda.synchronize()
    m = hidden.size(0)
    return (buf.x[:m].view(torch.uint8).clone(), buf.x_sf[:m].clone(),
            buf.topk_idx[:m].clone(), buf.topk_weights[:m].clone())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()
    buf_old, buf_new = make_buffer(64), make_buffer(64)
    names = ("x", "x_sf", "topk_idx", "topk_weights")
    mismatches, total = 0, 0
    for seed in range(args.seeds):
        torch.manual_seed(1000 + seed)
        w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05)
        if seed % 2 == 1:   # coarse weights/activations -> many exact logit ties
            w = (w * 8).round().to(torch.bfloat16) / 8
        for m in args.rows:
            hidden = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            hidden *= torch.exp(torch.randn(m, 1, device="cuda")).to(torch.bfloat16)
            if seed % 2 == 1:
                hidden = (hidden * 4).round().to(torch.bfloat16) / 4
            for quant in ("mxfp4", "qoq"):
                old = run(buf_old, hidden, w, quant, 0)
                new = run(buf_new, hidden, w, quant, 1)
                total += 1
                bad = [n for n, a, b in zip(names, old, new) if not torch.equal(a, b)]
                if bad:
                    mismatches += 1
                    print(f"MISMATCH seed={seed} m={m} quant={quant}: {bad}")
                    if "topk_idx" in bad:
                        print("  old idx", old[2].tolist()); print("  new idx", new[2].tolist())
    print(f"frontend tiny-M bit-identity: {total - mismatches}/{total} identical "
          f"(seeds={args.seeds}, rows={args.rows}, quant=mxfp4+qoq)")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
