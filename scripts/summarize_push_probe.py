#!/usr/bin/env python3
"""Summarize run_probe.sh logs for the push-dispatch A/B (PUSH ON vs OFF).

Usage: python3 scripts/summarize_push_probe.py [tag]   (reads probe_{quant}_m{M}_push{ON,OFF}{tag}{rep}.log)
Prints rank-0 medians (us from kernel entry) of slots 1 (barrier #1 done), 2 (pull done /
pool ready), 3 (first math), 7 (kernel end), 16 (barrier-1 wait incl. cross-rank launch
skew) and the skew-corrected kernel end (7 - 16).
"""
import glob, re, sys

tag = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for path in sorted(glob.glob(f"probe_*_m*_push*{tag}*.log")):
    m = re.match(r"probe_(\w+?)_m(\d+)_push(ON|OFF)" + re.escape(tag) + r"(\d+)\.log", path.split("/")[-1])
    if not m:
        continue
    quant, M, mode, rep = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    med = {}
    for line in open(path):
        mm = re.match(r"\s*(\d+)\s+(.+?)\s{2,}([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if mm:
            med[int(mm.group(1))] = float(mm.group(3))
    rows.append((quant, M, mode, rep, med))
print(f"{'quant':6} {'M':>3} {'push':4} rep {'s1':>7} {'s2':>7} {'s3':>7} {'s7':>7} {'s16':>7} {'end-skew':>8}")
for quant, M, mode, rep, med in sorted(rows, key=lambda r: (r[0], r[1], r[2] != "ON", r[3])):
    if not all(s in med for s in (1, 2, 3, 7, 16)):
        print(f"{quant:6} {M:3d} {mode:4} {rep}   (incomplete: {sorted(med)[:5]}...)")
        continue
    print(f"{quant:6} {M:3d} {mode:4} {rep}   {med[1]:7.1f} {med[2]:7.1f} {med[3]:7.1f} {med[7]:7.1f} {med[16]:7.1f} {med[7]-med[16]:8.1f}")
