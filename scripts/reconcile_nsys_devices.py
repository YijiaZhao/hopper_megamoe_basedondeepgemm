#!/usr/bin/env python3
"""Per-device fused-kernel durations from nsys reports (official capture or probe-under-nsys).

For every *.nsys-rep given: export sqlite, take graph-node kernels whose name contains
"mega_moe_h20_fused", align launches by index from the end across the 8 devices and print
per-round start skew, GPU0 duration, min/max-over-devices duration, plus medians over the
official window (last 3) and over the last N rounds (--last, default 20 = probe iters).
"""
import argparse
import pathlib
import sqlite3
import statistics
import subprocess


def load(report):
    db = pathlib.Path("/tmp") / f"{report.stem}.reconcile.sqlite"
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
                    "--output", str(db), str(report)], check=True, capture_output=True)
    con = sqlite3.connect(db)
    rows = con.execute(
        """select k.deviceId, s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on s.id = k.shortName
           where k.graphNodeId is not null and s.value like '%mega_moe_h20_fused%'
           order by k.deviceId, k.start""").fetchall()
    con.close()
    per_dev, names = {}, set()
    for dev, name, start, end in rows:
        per_dev.setdefault(dev, []).append((start, end))
        names.add(name)
    return per_dev, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=pathlib.Path)
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--per-round", action="store_true")
    args = ap.parse_args()
    for report in args.reports:
        per_dev, names = load(report)
        devs = sorted(per_dev)
        n = min(len(per_dev[d]) for d in devs)
        print(f"\n## {report.name}: kernels={sorted(names)} launches/device="
              f"{[len(per_dev[d]) for d in devs]}")
        rounds = []
        for r in range(n):
            ev = {d: per_dev[d][len(per_dev[d]) - n + r] for d in devs}
            starts = [ev[d][0] for d in devs]
            durs = {d: (ev[d][1] - ev[d][0]) / 1e3 for d in devs}
            rounds.append(dict(idx=r, skew=(max(starts) - min(starts)) / 1e3,
                               gpu0=durs[devs[0]], dmin=min(durs.values()),
                               dmax=max(durs.values()),
                               end_skew=(max(e for _, e in ev.values()) - min(e for _, e in ev.values())) / 1e3,
                               durs=durs))
        if args.per_round:
            print(" round  skew  GPU0dur  mindur  maxdur  endskew | per-device dur")
            for x in rounds:
                print(f"  {x['idx']:>3} {x['skew']:6.1f} {x['gpu0']:8.1f} {x['dmin']:7.1f} {x['dmax']:7.1f} "
                      f"{x['end_skew']:7.1f} | " + " ".join(f"{x['durs'][d]:6.1f}" for d in devs))

        def med(sel, key):
            return statistics.median(x[key] for x in sel)
        for label, sel in (("last3 (official window)", rounds[-3:]),
                           (f"last{args.last}", rounds[-args.last:]), ("all", rounds)):
            print(f"  {label:<24} GPU0 dur {med(sel,'gpu0'):6.1f}  min-over-dev {med(sel,'dmin'):6.1f}  "
                  f"max-over-dev {med(sel,'dmax'):6.1f}  start skew {med(sel,'skew'):5.1f}  "
                  f"end skew {med(sel,'end_skew'):4.1f}  (n={len(sel)})")


if __name__ == "__main__":
    main()
