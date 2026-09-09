#!/usr/bin/env python3
"""Summarize run_probe.sh logs for the tiny-M GEMV A/B (TINYM ON vs OFF).

Usage: python3 scripts/summarize_tinym_probe.py [tag]   (reads probe_{quant}_m{M}_tinym{ON,OFF}{tag}{rep}.log)
Prints rank-0 medians of slots 3/4/5/7 (us from kernel entry), slot 16 (barrier-1 wait incl.
cross-rank skew), the skew-corrected kernel end (7 - 16), and the achieved weight-streaming
bandwidth per phase: L1 = M x 10 tiles x 24 K-blocks x 20480 B / (slot4 - slot3),
L2 = M x 12 x 10 x 20480 B / (slot5 - slot4) (the probe routes one row per global token to
every rank, one expert per row, so a rank's active pool blocks == M).
"""
import glob, re, sys

tag = sys.argv[1] if len(sys.argv) > 1 else ""
TILE = 20480
rows = []
for path in sorted(glob.glob(f"probe_*_m*_tinym*{tag}*.log")):
    m = re.match(r"probe_(\w+?)_m(\d+)_tinym(ON|OFF)" + re.escape(tag) + r"(\d+)\.log", path.split("/")[-1])
    if not m:
        continue
    quant, M, mode, rep = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    med = {}
    for line in open(path):
        mm = re.match(r"\s*(\d+)\s+(.+?)\s{2,}([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if mm:
            med[int(mm.group(1))] = float(mm.group(3))
    if not all(s in med for s in (3, 4, 5, 7)):
        rows.append((quant, M, mode, rep, None))
        continue
    l1 = M * 10 * 24 * TILE / max(med[4] - med[3], 1e-9) / 1e3   # GB/s (bytes / us / 1e3)
    l2 = M * 12 * 10 * TILE / max(med[5] - med[4], 1e-9) / 1e3
    rows.append((quant, M, mode, rep, (med[3], med[4], med[5], med[7], med.get(16, float("nan")), l1, l2)))

print(f"{'quant':<6}{'M':>3} {'mode':<4}{'rep':>3} {'s3':>8} {'s4':>8} {'s5':>8} {'s7':>8} {'s16':>8} {'s7-s16':>8} {'L1GB/s':>8} {'L2GB/s':>8}")
for quant, M, mode, rep, v in rows:
    if v is None:
        print(f"{quant:<6}{M:>3} {mode:<4}{rep:>3}  (no stamps: run failed?)")
        continue
    s3, s4, s5, s7, s16, l1, l2 = v
    print(f"{quant:<6}{M:>3} {mode:<4}{rep:>3} {s3:8.2f} {s4:8.2f} {s5:8.2f} {s7:8.2f} {s16:8.2f} {s7 - s16:8.2f} {l1:8.0f} {l2:8.0f}")
