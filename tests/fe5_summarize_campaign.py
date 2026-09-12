#!/usr/bin/env python3
"""Aggregate a tests/fe5_campaign*.sh run: per (routing mode, quant, M) the median over passes of each capture's
GPU-0 median-of-last-3 spans (TIMELINE_LAST3.json written by scripts/summarize_four_api_h20_last3.py).
Usage: python3 tests/fe5_summarize_campaign.py /raid/kimi/results/fe5/cap [skew_limit_us=20]"""
import json, pathlib, statistics, sys, re
root = pathlib.Path(sys.argv[1])
SKEW_LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
cells = {}   # (mode, quant, M) -> {"fe": [...], "mega": [...], "e2e": [...], "skew": [...], "e2e_f": [...], "mega_f": [...]}
mega_only = {}
for d in sorted(p for p in root.glob("*_p*") if p.is_dir()):
    js = d / "TIMELINE_LAST3.json"
    if not js.exists():
        print(f"missing {js}", file=sys.stderr); continue
    m = re.fullmatch(r"(?:\w+_)?(normal|balanced)_p(\d+)|(megaonly)_p(\d+)", d.name)
    if m is None:
        continue
    for row in json.load(open(js)):
        if row["backend"] != "fused":
            continue
        if m.group(3):
            mega_only.setdefault((row["precision"], row["M"]), []).append(row["mega_median_us"])
            continue
        c = cells.setdefault((m.group(1), row["precision"], row["M"]), {"fe": [], "mega": [], "e2e": [], "skew": [], "e2e_f": [], "mega_f": []})
        c["fe"].append(row["frontend_median_us"]); c["mega"].append(row["mega_median_us"]); c["e2e"].append(row["target_median_us"])
        sk = row.get("mega_start_skew_max_us")
        c["skew"].append(sk)
        # skew filter: keep a pass only if none of its last-3 replays had an inter-rank Mega-start skew above SKEW_LIMIT us
        if sk is not None and sk <= SKEW_LIMIT:
            c["e2e_f"].append(row["target_median_us"]); c["mega_f"].append(row["mega_median_us"])
def med(v): return f"{statistics.median(v):.1f}" if v else "-"
EMPTY = {"fe": [], "mega": [], "e2e": [], "skew": [], "e2e_f": [], "mega_f": []}
print("| Precision | M | routing | FE | E2E Mega | E2E | Mega-only | passes (e2e/mega-only) | "
      f"Mega-start skew (median of per-pass max, us) | E2E, passes with skew <= {SKEW_LIMIT:.0f} us (n) |")
print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
for q in ("mxfp4", "qoq"):
    for M in (2, 4, 8, 16):
        for mode in ("normal", "balanced"):
            c = cells.get((mode, q, M), EMPTY)
            sk = [v for v in c["skew"] if v is not None]
            print(f"| {q.upper()} | {M} | {mode} | {med(c['fe'])} | {med(c['mega'])} | **{med(c['e2e'])}** | "
                  f"{med(mega_only.get((q, M), []))} | {len(c['fe'])}/{len(mega_only.get((q, M), []))} | "
                  f"{med(sk)} | {med(c['e2e_f'])} ({len(c['e2e_f'])}) |")
print("\nPer-pass FE spans (us):")
for k, c in sorted(cells.items()):
    print(f"  {k[0]:8s} {k[1]:5s} M{k[2]:<3d} {['%.1f' % v for v in c['fe']]}")
