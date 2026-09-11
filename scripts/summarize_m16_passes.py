#!/usr/bin/env python3
"""Median across independent capture passes of the customer number (GPU0, last-3 median)
plus per-report start skew (max - min device start of the last-3 rounds).

usage: summarize_m16_passes.py <pass_dir>... [--skew-us 20]
Each pass dir is one capture_four_api_h20_timelines.sh OUT (TIMELINE_LAST3.json + *.nsys-rep).
"""
import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reconcile_nsys_devices import load  # noqa: E402
from summarize_four_api_h20_timelines import extract_final_three  # noqa: E402


def report_skew(report):
    per_dev, _ = load(report)
    devs = sorted(per_dev)
    n = min(len(per_dev[d]) for d in devs)
    skews, dmins = [], []
    for r in range(max(0, n - 3), n):
        ev = {d: per_dev[d][len(per_dev[d]) - n + r] for d in devs}
        starts = [ev[d][0] for d in devs]
        skews.append((max(starts) - min(starts)) / 1e3)
        dmins.append(min((e - s) / 1e3 for s, e in ev.values()))
    return (statistics.median(skews) if skews else None,
            max(skews) if skews else None,
            statistics.median(dmins) if dmins else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("passes", nargs="+", type=pathlib.Path)
    ap.add_argument("--skew-us", type=float, default=20.0)
    args = ap.parse_args()

    per_key = {}      # (scope, quant, M, backend) -> list of (pass, gpu0_target, mega, skew_med, skew_max, dmin)
    n_reports = n_skewed = 0
    for pdir in args.passes:
        reports = sorted(pdir.glob("*.nsys-rep"))
        if not reports:
            print(f"(skip {pdir}: no reports)", file=sys.stderr)
            continue
        for rep in reports:
            # GPU0 last-3 medians straight from the report (sub-matrix captures have no
            # TIMELINE_LAST3.json: the matrix summarizers require the full 2x2x2 matrix).
            metadata, final_three = extract_final_three(rep, 0)
            key = (metadata["scope"], metadata["quant"], metadata["M"], metadata["backend"])
            target = statistics.median(c["target_span_us"] for c in final_three)
            mega = statistics.median(c["mega_span_us"] for c in final_three)
            skew_med = skew_max = dmin = None
            if metadata["backend"] == "fused":
                try:
                    skew_med, skew_max, dmin = report_skew(rep)
                except Exception as exc:  # noqa: BLE001
                    print(f"(skew failed {rep}: {exc})", file=sys.stderr)
                n_reports += 1
                if skew_max is not None and skew_max > args.skew_us:
                    n_skewed += 1
            per_key.setdefault(key, []).append((pdir.name, target, mega, skew_med, skew_max, dmin))

    def med(vals):
        vals = [v for v in vals if v is not None]
        return f"{statistics.median(vals):.1f}" if vals else "-"

    print(f"Passes: {[p.name for p in args.passes]}  (customer number = GPU0 median of last 3; "
          f"'med' = median across passes; skew = max start skew of the last-3 rounds)\n")
    print("| scope | quant | M | backend | target med | per pass | mega med | per pass | skew max/pass (us) | min-over-dev med |")
    print("|---|---|---:|---|---:|---|---:|---|---|---:|")
    for key in sorted(per_key, key=lambda k: ({"e2e": 0, "mega": 1}[k[0]], k[1], k[2], k[3])):
        rows = per_key[key]
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {med([r[1] for r in rows])} | "
              f"{' / '.join(f'{r[1]:.1f}' for r in rows)} | {med([r[2] for r in rows])} | "
              f"{' / '.join(f'{r[2]:.1f}' for r in rows)} | "
              f"{' / '.join('-' if r[4] is None else f'{r[4]:.1f}' for r in rows)} | "
              f"{med([r[5] for r in rows])} |")
    print(f"\nFused captures with start skew > {args.skew_us:g} us: {n_skewed} / {n_reports}")


if __name__ == "__main__":
    main()
