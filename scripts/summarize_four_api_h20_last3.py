#!/usr/bin/env python3
"""List the final three target executions for each H20 timeline."""
import argparse
import csv
import json
import pathlib
import statistics

from summarize_four_api_h20_timelines import extract_final_three


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for report in sorted(args.directory.glob("*.nsys-rep")):
        metadata, final_three = extract_final_three(report, args.device_id)
        target_values = [call["target_span_us"] for call in final_three]
        frontend_values = [call["frontend_us"] for call in final_three]
        mega_values = [call["mega_span_us"] for call in final_three]
        rows.append(dict(
            scope=metadata["scope"],
            precision=metadata["quant"],
            M=metadata["M"],
            backend=metadata["backend"],
            device_id=args.device_id,
            mega_third_last_us=mega_values[0],
            mega_second_last_us=mega_values[1],
            mega_last_us=mega_values[2],
            mega_median_us=statistics.median(mega_values),
            frontend_median_us=None if frontend_values[0] is None else statistics.median(frontend_values),
            target_median_us=statistics.median(target_values),
            report=metadata["report"],
        ))

    if len(rows) != 24:
        raise RuntimeError(f"expected 24 reports, found {len(rows)}")
    order_scope = {"e2e": 0, "mega": 1}
    order_quant = {"mxfp4": 0, "qoq": 1}
    order_backend = {"fused": 0, "split": 1}
    rows.sort(key=lambda row: (
        order_scope[row["scope"]], order_quant[row["precision"]],
        row["M"], order_backend[row["backend"]],
    ))

    with open(args.directory / "TIMELINE_LAST3.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.directory / "TIMELINE_LAST3.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        f"Device: GPU {args.device_id}; final value: median of the three displayed executions.\n",
        "| Scope | Precision | M | Backend | Mega -3 | Mega -2 | Mega last | Mega median | Frontend median | Target median |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope'].upper()} | {row['precision'].upper()} | {row['M']} | "
            f"{row['backend'].upper()} | {fmt(row['mega_third_last_us'])} | "
            f"{fmt(row['mega_second_last_us'])} | {fmt(row['mega_last_us'])} | "
            f"**{fmt(row['mega_median_us'])}** | {fmt(row['frontend_median_us'])} | "
            f"{fmt(row['target_median_us'])} |"
        )
    text = "\n".join(lines) + "\n"
    (args.directory / "TIMELINE_LAST3.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
