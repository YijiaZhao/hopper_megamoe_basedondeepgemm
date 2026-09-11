#!/usr/bin/env python3
"""Median / min / max across independent official captures of the customer number.

Each capture directory holds TIMELINE_LAST3.csv (scripts/summarize_four_api_h20_last3.py:
GPU0 median of the last 3 spans) and optionally reconcile_fused.txt (start skew per
fused report). For every (scope, quant, M, backend) the customer number is aggregated
over the captures of each knob value: median, min, max, n, and the number of captures
whose fused report had a last-3 start skew > SKEW_US (reported, not excluded).

Usage: summarize_knob_captures.py --knob 0 DIR... --knob 1 DIR... [--fused-only] [--skew-us 20]
"""
import argparse
import csv
import os
import re
import statistics
import sys


def read_last3(d):
    rows = {}
    with open(os.path.join(d, "TIMELINE_LAST3.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["scope"], r["precision"], int(r["M"]), r["backend"])
            rows[key] = (float(r["target_median_us"]), r["report"])
    return rows


def read_skew(d):
    skew = {}
    p = os.path.join(d, "reconcile_fused.txt")
    if not os.path.exists(p):
        return skew
    rep = None
    for line in open(p):
        m = re.match(r"## (\S+\.nsys-rep)", line)
        if m:
            rep = m.group(1)
            continue
        m = re.search(r"last3 \(official window\).*start skew\s+([0-9.]+)", line)
        if m and rep:
            skew[rep] = float(m.group(1))
    return skew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", action="append", nargs="+", required=True,
                    help="knob value followed by its capture dirs")
    ap.add_argument("--fused-only", action="store_true")
    ap.add_argument("--skew-us", type=float, default=20.0)
    a = ap.parse_args()
    groups = {g[0]: g[1:] for g in a.knob}
    data = {}  # key -> knob -> list of (value, skew)
    for knob, dirs in groups.items():
        for d in dirs:
            rows, skew = read_last3(d), read_skew(d)
            for key, (v, rep) in rows.items():
                if a.fused_only and key[3] != "fused":
                    continue
                data.setdefault(key, {}).setdefault(knob, []).append((v, skew.get(rep)))
    knobs = list(groups)
    hdr = ["scope", "quant", "M", "backend"]
    for k in knobs:
        hdr += [f"k{k}_median", f"k{k}_min", f"k{k}_max", f"k{k}_n", f"k{k}_skew>{a.skew_us:g}"]
    if len(knobs) == 2:
        hdr.append(f"k{knobs[1]}-k{knobs[0]}_median")
    print(",".join(hdr))
    for key in sorted(data, key=lambda k: (k[0], k[1], k[2], k[3])):
        line = [key[0], key[1], str(key[2]), key[3]]
        meds = []
        for k in knobs:
            vals = data[key].get(k, [])
            if not vals:
                line += ["", "", "", "0", ""]
                meds.append(None)
                continue
            xs = [v for v, _ in vals]
            med = statistics.median(xs)
            meds.append(med)
            nsk = sum(1 for _, s in vals if s is not None and s > a.skew_us)
            line += [f"{med:.2f}", f"{min(xs):.2f}", f"{max(xs):.2f}", str(len(xs)), str(nsk)]
        if len(knobs) == 2 and None not in meds:
            line.append(f"{meds[1] - meds[0]:+.2f}")
        print(",".join(line))


if __name__ == "__main__":
    sys.exit(main())
