"""Per-CTA timeline analysis of the fused MegaMoE task log dumped by tests/profile_fused_phase_stamps.py
(PROBE_TASKLOG=1 PROBE_DUMP_DIR=<dir>): for one rank and iteration, where the 78 CTAs idle
(start delay, gaps between tasks, wait for the kernel-wide last task), per-phase work totals and the
ideal (perfectly balanced) busy time.

  python3 scripts/analyze_probe_tasklog.py <dir>/stamps_mxfp4_m2_rank0.pt [--iter k] [--chains 10]
"""
import argparse
import statistics

import torch

BASE, PER_CTA, MAX_SMS = 64, 16, 160


def parse(s):
    t0 = s[0]
    ctas = []
    for sm in range(MAX_SMS):
        base = BASE + sm * PER_CTA
        count = s[base + 15]
        if count <= 0:
            continue
        entry = s[base + 14]
        tasks = []
        for t in range(min(count, 7)):
            w0, w1 = s[base + 2 * t], s[base + 2 * t + 1]
            meta = (w0 >> 32) & 0xffffffff
            tasks.append(dict(l2=bool(meta >> 31 & 1), pb=(meta >> 24) & 0x7f, nb=(meta >> 16) & 0xff,
                              ks=(meta >> 8) & 0xff, nks=meta & 0xff,
                              start=(entry + (w0 & 0xffffffff) - t0) / 1000.0, end=(entry + w1 - t0) / 1000.0))
        ctas.append((sm, count, tasks))
    stamps = {k: (s[k] - t0) / 1000.0 for k in (1, 3, 4, 5, 7)}
    return ctas, stamps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--iter", type=int, default=-1)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--block", type=int, default=-1, help="also list every task of this pool block (start order)")
    ap.add_argument("--late", type=float, default=0.0, help="list every task ending in the last N us before last-L2")
    a = ap.parse_args()
    d = torch.load(a.path)
    raw = d["raw"][a.iter]
    ctas, st = parse(raw)
    kend = st[7]
    print(f"{a.path} iter {a.iter}: routing-done {st[1]:.1f} first-math {st[3]:.1f} last-L1 {st[4]:.1f} "
          f"last-L2 {st[5]:.1f} kernel-end {kend:.1f} us; {len(ctas)} CTAs, truncated CTAs (>7 tasks): "
          f"{sum(1 for c in ctas if c[1] > 7)}")
    l1 = [t for c in ctas for t in c[2] if not t["l2"]]
    l2 = [t for c in ctas for t in c[2] if t["l2"]]
    busy_l1 = sum(t["end"] - t["start"] for t in l1)
    busy_l2 = sum(t["end"] - t["start"] for t in l2)
    n = len(ctas)
    first_math = min(t["start"] for c in ctas for t in c[2])
    print(f"tasks: L1 {len(l1)} (full {sum(1 for t in l1 if t['nks'] == 1)}, split {sum(1 for t in l1 if t['nks'] > 1)}) "
          f"L2 {len(l2)} (full {sum(1 for t in l2 if t['nks'] == 1)}, split {sum(1 for t in l2 if t['nks'] > 1)})")
    full_l1 = [t["end"] - t["start"] for t in l1 if t["nks"] == 1]
    full_l2 = [t["end"] - t["start"] for t in l2 if t["nks"] == 1]
    if full_l1:
        print(f"full L1 task us: p10 {sorted(full_l1)[len(full_l1)//10]:.1f} p50 {statistics.median(full_l1):.1f} "
              f"p90 {sorted(full_l1)[len(full_l1)*9//10]:.1f} max {max(full_l1):.1f}")
    if full_l2:
        print(f"full L2 task us: p10 {sorted(full_l2)[len(full_l2)//10]:.1f} p50 {statistics.median(full_l2):.1f} "
              f"p90 {sorted(full_l2)[len(full_l2)*9//10]:.1f} max {max(full_l2):.1f}")
    print(f"busy SM-us: L1 {busy_l1:.0f} L2 {busy_l2:.0f} total {busy_l1 + busy_l2:.0f} -> ideal per-SM busy "
          f"{(busy_l1 + busy_l2) / n:.1f} us; span first-math..last-L2 {st[5] - first_math:.1f} us; "
          f"balance loss (span - ideal) {st[5] - first_math - (busy_l1 + busy_l2) / n:.1f} us")
    # idle decomposition per CTA
    start_idle, gap_idle, end_idle = [], [], []
    for sm, count, tasks in ctas:
        start_idle.append(tasks[0]["start"] - first_math)
        gap_idle.append(sum(max(0.0, tasks[i + 1]["start"] - tasks[i]["end"]) for i in range(len(tasks) - 1)))
        end_idle.append(max(0.0, st[5] - tasks[-1]["end"]))
    print(f"idle per CTA (us): start-delay avg {statistics.mean(start_idle):.1f} max {max(start_idle):.1f} | "
          f"inter-task gaps avg {statistics.mean(gap_idle):.1f} max {max(gap_idle):.1f} | "
          f"after own last task until last-L2 avg {statistics.mean(end_idle):.1f} max {max(end_idle):.1f} "
          f"(CTAs idle > 5 us at the end: {sum(1 for v in end_idle if v > 5)})")
    # wave structure: L1 end times per CTA
    l1_end = sorted(max([t["end"] for t in c[2] if not t["l2"]] or [0.0]) for c in ctas)
    print(f"per-CTA last L1 end: p0 {l1_end[0]:.1f} p50 {l1_end[len(l1_end)//2]:.1f} p90 {l1_end[len(l1_end)*9//10]:.1f} "
          f"p100 {l1_end[-1]:.1f}")
    # end-of-kernel: which tasks end in the last 8 us
    late = sorted([(t["end"], t["start"], c[0], t) for c in ctas for t in c[2] if t["end"] > st[5] - 8.0], reverse=True)
    print(f"tasks ending in the last 8 us before last-L2: {len(late)} "
          f"(L2 {sum(1 for x in late if x[3]['l2'])}, L1 {sum(1 for x in late if not x[3]['l2'])})")
    def fmt(t):
        return (f"{'L2' if t['l2'] else 'L1'}[b{t['pb']},n{t['nb']}"
                f"{('/' + str(t['ks']) + 'of' + str(t['nks'])) if t['nks'] > 1 else ''}]{t['start']:.1f}-{t['end']:.1f}")
    if a.block >= 0:
        sel = sorted([(t["start"], c[0], t) for c in ctas for t in c[2] if t["pb"] == a.block])
        print(f"tasks of pool block {a.block} ({len(sel)}):")
        for st_, sm, t in sel:
            print(f"  sm{sm:3d} {fmt(t)} dur {t['end'] - t['start']:.1f}")
    if a.late > 0:
        sel = sorted([(t["end"], c[0], t) for c in ctas for t in c[2] if t["end"] > st[5] - a.late])
        print(f"tasks ending in the last {a.late} us ({len(sel)}):")
        for e, sm, t in sel:
            print(f"  sm{sm:3d} {fmt(t)} dur {t['end'] - t['start']:.1f}")
    ctas.sort(key=lambda c: -(c[2][-1]["end"] if c[2] else 0.0))
    for sm, count, tasks in ctas[:a.chains]:
        print(f"  sm{sm:3d} ({count}) " + " ".join(
            f"{'L2' if t['l2'] else 'L1'}[b{t['pb']},n{t['nb']}{('/' + str(t['ks']) + 'of' + str(t['nks'])) if t['nks'] > 1 else ''}]"
            f"{t['start']:.1f}-{t['end']:.1f}" for t in tasks))


if __name__ == "__main__":
    main()
