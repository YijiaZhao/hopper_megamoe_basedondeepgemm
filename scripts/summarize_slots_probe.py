#!/usr/bin/env python3
"""Summarize a run_pdf_ab.sh MODE=perf log (probe under nsys, knob A/B) into one table.

Per "--- PROBE+NSYS {quant}_M{M}_{tag}_p{pass}" section: rank-0 median (us from kernel entry)
of slots 8 (routing atomics), 9 (push issued), 1 (DONE / barrier #1), 2 (pool ready),
3 (first math), 7 (kernel end), 16 (barrier-1 wait incl. skew), the skew-corrected end
(7 - 16), and the nsys min-over-devices kernel duration (median of the last 20 launches).
Usage: summarize_slots_probe.py pdf_perf.log [--knobs k1,k0]
"""
import re, sys, statistics

log = open(sys.argv[1]).read()
sections = re.split(r"^--- PROBE\+NSYS ", log, flags=re.M)[1:]
rows = {}
for sec in sections:
    head = sec.split("\n", 1)[0]
    m = re.match(r"(\w+?)_M(\d+)_(\w+)_p(\d+)", head)
    if not m:
        continue
    q, M, tag, p = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    slots = {}
    for sm in re.finditer(r"^\s+(\d+) (.+?)\s{2,}(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", sec, flags=re.M):
        slots.setdefault(int(sm.group(1)), float(sm.group(3)))
    rec = re.search(r"last20\s+GPU0 dur\s+([\d.]+)\s+min-over-dev\s+([\d.]+)", sec)
    rows[(q, M, tag, p)] = (slots, rec)

def cell(slots, i):
    return f"{slots[i]:6.2f}" if i in slots else "   n/a"

print(f"{'point':<14}{'tag':<5}{'p':<3}{'s8':>7}{'s9':>7}{'s1':>7}{'s2':>7}{'s3':>7}{'s7':>7}{'s16':>7}{'7-16':>7}"
      f"{'nsys GPU0':>10}{'min-dev':>9}")
for key in sorted(rows, key=lambda k: (k[0], k[1], k[3], k[2])):
    q, M, tag, p = key
    slots, rec = rows[key]
    end_corr = f"{slots[7] - slots[16]:6.2f}" if 7 in slots and 16 in slots else "   n/a"
    g0 = f"{float(rec.group(1)):9.1f}" if rec else "      n/a"
    mn = f"{float(rec.group(2)):8.1f}" if rec else "     n/a"
    print(f"{q + ' M' + str(M):<14}{tag:<5}{p:<3}{cell(slots,8)}{cell(slots,9)}{cell(slots,1)}{cell(slots,2)}"
          f"{cell(slots,3)}{cell(slots,7)}{cell(slots,16)}{end_corr} {g0} {mn}")
