#!/usr/bin/env python3
"""Aggregate a tests/fe5_campaign.sh run: per (FE build, routing mode, quant, M) the median over passes of each
capture's GPU-0 median-of-last-3 spans (TIMELINE_LAST3.json written by scripts/summarize_four_api_h20_last3.py).
Usage: python3 tests/fe5_summarize_campaign.py /raid/kimi/results/fe5/cap"""
import json, pathlib, statistics, sys, re
root = pathlib.Path(sys.argv[1])
SKEW_LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
cells = {}   # (cfg, mode, quant, M) -> {"fe": [...], "mega": [...], "e2e": [...]}
mega_only = {}
for d in sorted(root.glob("*_p*")):
    js = d / "TIMELINE_LAST3.json"
    if not js.exists():
        print(f"missing {js}", file=sys.stderr); continue
    m = re.fullmatch(r"(base|lean)_(normal|balanced)_p(\d+)|(megaonly)_p(\d+)", d.name)
    for row in json.load(open(js)):
        if row["backend"] != "fused":
            continue
        if m.group(4):
            mega_only.setdefault((row["precision"], row["M"]), []).append(row["mega_median_us"])
            continue
        c = cells.setdefault((m.group(1), m.group(2), row["precision"], row["M"]), {"fe": [], "mega": [], "e2e": [], "skew": [], "e2e_f": [], "mega_f": []})
        c["fe"].append(row["frontend_median_us"]); c["mega"].append(row["mega_median_us"]); c["e2e"].append(row["target_median_us"])
        sk = row.get("mega_start_skew_max_us")
        c["skew"].append(sk)
        # skew filter: keep a pass only if none of its last-3 replays had an inter-rank Mega-start skew above SKEW_LIMIT us
        if sk is not None and sk <= SKEW_LIMIT:
            c["e2e_f"].append(row["target_median_us"]); c["mega_f"].append(row["mega_median_us"])
def med(v): return f"{statistics.median(v):.1f}" if v else "-"
def rng(v): return f"{min(v):.1f}-{max(v):.1f}" if v else "-"
EMPTY = {"fe": [], "mega": [], "e2e": [], "skew": [], "e2e_f": [], "mega_f": []}
print("| Precision | M | routing | FE base | FE lean | E2E Mega base | E2E Mega lean | E2E base | E2E lean | Mega-only | passes (base/lean/mega-only) | "
      f"Mega-start skew base / lean (median of per-pass max, us) | E2E base / lean, passes with skew <= {SKEW_LIMIT:.0f} us (n) |")
print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for q in ("mxfp4", "qoq"):
    for M in (2, 4, 8, 16):
        for mode in ("normal", "balanced"):
            b = cells.get(("base", mode, q, M), EMPTY); l = cells.get(("lean", mode, q, M), EMPTY)
            skb = [v for v in b["skew"] if v is not None]; skl = [v for v in l["skew"] if v is not None]
            print(f"| {q.upper()} | {M} | {mode} | {med(b['fe'])} | {med(l['fe'])} | {med(b['mega'])} | {med(l['mega'])} | "
                  f"**{med(b['e2e'])}** | **{med(l['e2e'])}** | {med(mega_only.get((q, M), []))} | {len(b['fe'])}/{len(l['fe'])}/{len(mega_only.get((q, M), []))} | "
                  f"{med(skb)} / {med(skl)} | {med(b['e2e_f'])} ({len(b['e2e_f'])}) / {med(l['e2e_f'])} ({len(l['e2e_f'])}) |")
print("\nPer-pass FE spans (us), base -> lean:")
for k, c in sorted(cells.items()):
    if k[0] == "base":
        l = cells.get(("lean",) + k[1:], {"fe": []})
        print(f"  {k[1]:8s} {k[2]:5s} M{k[3]:<3d} base {['%.1f' % v for v in c['fe']]} lean {['%.1f' % v for v in l['fe']]}")
