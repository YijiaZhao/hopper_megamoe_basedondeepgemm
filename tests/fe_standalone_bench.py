"""Single-GPU, single-process Fable frontend timing (no torchrun / MegaMoE): CUDA-event
FE time over N eager launches, each after a 256 MB L2 flush (same flush as the 8-rank
bench), plus the DG_FE_STAMPS phase attribution of 5 stamped launches. Meant to run one
configuration per GPU in parallel (CUDA_VISIBLE_DEVICES=k).

  DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=fma python3 tests/fe_standalone_bench.py --quant mxfp4 --rows 1
"""
import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import deep_gemm  # noqa: E402
from ncu_frontend_tinym import HIDDEN, EXPERTS, TOPK, make_buffer  # noqa: E402
from bench_frontend_tinym import attribution, fmt_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=("mxfp4", "qoq"), required=True)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grid", default=os.environ.get("DG_FE_TINYM_GRID", "96"))
    ap.add_argument("--mma", default=os.environ.get("DG_FE_TINYM_MMA", "wmma"))
    args = ap.parse_args()
    torch.manual_seed(20260805)
    w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    torch.manual_seed(17000)
    x = torch.randn(args.rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    buf = make_buffer(64)
    scratch = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
    tinym = int(os.environ.get("DG_FE_TINYM", "1"))
    # DG_FE_ROUTER_WLAYOUT=fragment: permute ONCE here (weight-transform time) and pass wlayout="pre",
    # so the timed call has no wrapper-side lookup (the standalone event is CPU-launch-bound).
    wlayout = os.environ.get("DG_FE_ROUTER_WLAYOUT", "row")
    if wlayout == "fragment":
        w = deep_gemm.fable_router_weight_fragment_layout(w)
        wlayout = "pre"

    def fe(st=0):
        deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant=args.quant, tinym=tinym, stamps=st,
                                                   grid=args.grid, mma=args.mma, wlayout=wlayout)

    times = []
    for it in range(args.warmup + args.iters):
        scratch.zero_(); torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fe(); e1.record(); torch.cuda.synchronize()
        if it >= args.warmup:
            times.append(e0.elapsed_time(e1) * 1e3)
    n_router = deep_gemm.fable_frontend_router_ctas(args.rows, EXPERTS, HIDDEN, TOPK, tinym, args.grid)
    fullk = bool(tinym) and args.rows <= 16 and str(args.grid).strip().lower() != "96"
    print(f"== FE standalone eager (L2 flush before each launch): quant={args.quant} rows={args.rows} "
          f"DG_FE_TINYM_GRID={args.grid} DG_FE_TINYM_MMA={args.mma} DG_FE_ROUTER_WLAYOUT={os.environ.get('DG_FE_ROUTER_WLAYOUT', 'row')} DG_FE_TINYM_KPARTS={os.environ.get('DG_FE_TINYM_KPARTS', '1')} router CTAs={n_router} full-K={int(fullk)} "
          f"device={torch.cuda.get_device_name()} iters={args.iters} ==")
    print(fmt_stats("FE", times))
    collected = []
    for _ in range(5):
        scratch.zero_(); torch.cuda.synchronize()
        fe(1); torch.cuda.synchronize()
        collected.append(deep_gemm.fable_frontend_stamps(buf, EXPERTS))
    attribution(collected, n_router, args.rows, fullk)
    print(f"topk_idx[0]={buf.topk_idx[0].tolist()}")


if __name__ == "__main__":
    main()
