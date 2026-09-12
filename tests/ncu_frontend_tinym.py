#!/usr/bin/env python3
"""Standalone single-GPU driver for Nsight Compute profiling of the Fable frontend kernel
(router_cc_lean_kernel for rows <= 2, router_quant_topk_kernel otherwise; csrc/fable_frontend.cu).
No torch.distributed, no MegaMoE.

The kernel only sees rows_per_rank (the local M): with 8 ranks, global M = 2/4/8 map to
1 row on a rank and M = 16 maps to 2 rows, so this profiles rows in {1, 2}.

  ncu --kernel-name "regex:router_(quant_topk|cc_lean)_kernel" --launch-skip 5 --launch-count 1 --set full \
      --clock-control none -o <rep> python3 tests/ncu_frontend_tinym.py --quant mxfp4 --rows 1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=("mxfp4", "qoq"), required=True)
    ap.add_argument("--rows", type=int, default=1, help="rows per rank seen by the kernel (1 or 2)")
    ap.add_argument("--warmup", type=int, default=5, help="eager launches before the profiled one")
    ap.add_argument("--l2-flush", type=int, default=1, help="flush L2 (256 MB memset) before each launch")
    args = ap.parse_args()
    torch.manual_seed(20260805)
    w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    torch.manual_seed(17000)
    hidden = torch.randn(args.rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    buf = make_buffer(64)
    scratch = torch.empty(256 << 20, device="cuda", dtype=torch.uint8) if args.l2_flush else None
    for it in range(args.warmup + 1):  # the last launch is the one ncu profiles (--launch-skip warmup)
        if scratch is not None:
            scratch.zero_()
        deep_gemm.fable_router_quant_topk_frontend(hidden, w, buf, quant=args.quant, stamps=0)
        torch.cuda.synchronize()
    m = args.rows
    print(f"fe standalone: quant={args.quant} rows={m} path={deep_gemm.fable_frontend_path(m, HIDDEN, EXPERTS, TOPK)} launches={args.warmup + 1} "
          f"device={torch.cuda.get_device_name()} topk_idx[0]={buf.topk_idx[0].tolist()}")


if __name__ == "__main__":
    main()
