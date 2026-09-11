"""Reproducibility probe for the cc router (DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc): ONE
process = one fresh CUDA context; `--iters` stamped launches (each after a 256 MB L2 flush)
and the same number of unstamped CUDA-event launches. Prints one REPRO line:
kernel-end (merger slot 4 - earliest CTA start, us) median/min/p90, event time stamped and
unstamped median/min/p90. Driven by tests/run_fe_repro_cc.sh (10 processes x 4 cells).
"""
import argparse, os, sys, time, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import deep_gemm  # noqa: E402
from ncu_frontend_tinym import HIDDEN, EXPERTS, TOPK, make_buffer  # noqa: E402
from bench_frontend_tinym import pct, attribution  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=("mxfp4", "qoq"), required=True)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grid", default=os.environ.get("DG_FE_TINYM_GRID", "auto"))
    ap.add_argument("--mma", default=os.environ.get("DG_FE_TINYM_MMA", "cc"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--gap", choices=("compute", "clone", "sleep"), default="compute", help="between stamped launches: compute = read + reduce the stamps (D2H) per launch; clone = only clone the stamps, reduce at the end; sleep = clone + 5 ms host sleep")
    ap.add_argument("--attr", type=int, default=0, help="print the segment chain over the first N stamped launches")
    ap.add_argument("--order", choices=("st_first", "ev_first", "noev"), default="noev", help="noev = stamped launches without CUDA events around them (as fe_standalone_bench)")
    ap.add_argument("--wlayout", choices=("row", "fragment"), default="row", help="fragment = permute w like fe_standalone_bench (mma cc ignores the layout; timing probe)")
    args = ap.parse_args()
    torch.manual_seed(20260805)
    w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    torch.manual_seed(17000)
    x = torch.randn(args.rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    if args.wlayout == "fragment":
        w = deep_gemm.fable_router_weight_fragment_layout(w)
    buf = make_buffer(64)
    scratch = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)

    def fe(st):
        deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant=args.quant, tinym=1, stamps=st,
                                                   grid=args.grid, mma=args.mma)
    n_router = deep_gemm.fable_frontend_router_ctas(args.rows, EXPERTS, HIDDEN, TOPK, 1, args.grid)
    ends, ev_st, ev, collected, raws = [], [], [], [], []
    selmega = int(os.environ.get("DG_FE_SELECT_IN_MEGA", "0"))
    def kernel_end(raw):
        st = raw[: n_router + 1].double()
        t0 = st[:, 0].min()
        # kernel end: last-arriver "topk written" (merger row slot 4) or, DG_FE_SELECT_IN_MEGA=1, the last router CTA's "keys written" (slot 3)
        end = st[:n_router, 3].max() if selmega else st[n_router, 4]
        return float((end - t0) / 1e3)
    host_us = []
    def loop_ev():
        for it in range(args.warmup + args.iters):
            scratch.zero_(); torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            h0 = time.perf_counter(); e0.record(); fe(0); e1.record(); h1 = time.perf_counter(); torch.cuda.synchronize()
            if it >= args.warmup:
                ev.append(e0.elapsed_time(e1) * 1e3); host_us.append((h1 - h0) * 1e6)
    if args.order == "ev_first":
        loop_ev()
    for it in range(args.warmup + args.iters):
        scratch.zero_(); torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        if args.order == "noev":
            fe(1); torch.cuda.synchronize()
        else:
            e0.record(); fe(1); e1.record(); torch.cuda.synchronize()
        raw = deep_gemm.fable_frontend_stamps(buf, EXPERTS)
        if it >= args.warmup and len(collected) < args.attr: collected.append(raw)
        if it >= args.warmup:
            ev_st.append(e0.elapsed_time(e1) * 1e3 if args.order != "noev" else 0.0)
            if args.gap == "compute": ends.append(kernel_end(raw))
            else: raws.append(raw)
        if args.gap == "sleep": torch.cuda.synchronize(); time.sleep(0.005)
    for raw in raws: ends.append(kernel_end(raw))
    if args.order in ("st_first", "noev"):
        loop_ev()
    if collected: attribution(collected, n_router, args.rows, True)
    if args.iters <= 20: print("ends per launch:", " ".join(f"{v:.2f}" for v in ends))
    s = lambda v: f"{pct(v, 0.5):.2f} {min(v):.2f} {pct(v, 0.9):.2f}"
    print(f"REPRO {args.tag} order={args.order} gap={args.gap} wl={args.wlayout} selmega={selmega} quant={args.quant} rows={args.rows} pid={os.getpid()} n={args.iters} "
          f"end[med min p90]= {s(ends)} | ev_stamped= {s(ev_st)} | ev= {s(ev)} | host_launch_us= {pct(host_us, 0.5):.1f} | topk0={buf.topk_idx[0].tolist()[:3]}")


if __name__ == "__main__":
    main()
