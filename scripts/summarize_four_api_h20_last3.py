#!/usr/bin/env python3
"""List final-three Frontend, complete MegaMoE, and E2E spans per timeline."""
import argparse
import csv
import json
import pathlib
import statistics

from summarize_four_api_h20_timelines import extract_final_three


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def triplet_fields(prefix, values):
    if values[0] is None:
        return {
            f"{prefix}_third_last_us": None,
            f"{prefix}_second_last_us": None,
            f"{prefix}_last_us": None,
            f"{prefix}_median_us": None,
        }
    return {
        f"{prefix}_third_last_us": values[0],
        f"{prefix}_second_last_us": values[1],
        f"{prefix}_last_us": values[2],
        f"{prefix}_median_us": statistics.median(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for report in sorted(args.directory.glob("*.nsys-rep")):
        metadata, final_three = extract_final_three(report, args.device_id)
        frontend_values = [call["frontend_us"] for call in final_three]
        mega_values = [call["mega_span_us"] for call in final_three]
        target_values = [call["target_span_us"] for call in final_three]
        rows.append(dict(
            scope=metadata["scope"],
            precision=metadata["quant"],
            M=metadata["M"],
            backend=metadata["backend"],
            device_id=args.device_id,
            **triplet_fields("frontend", frontend_values),
            **triplet_fields("mega", mega_values),
            **triplet_fields("target", target_values),
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
        f"Device: GPU {args.device_id}; each median is taken over the three displayed complete spans.\n",
        "## E2E timelines",
        "",
        "| Precision | M | Backend | FE -3 | FE -2 | FE last | FE med | Mega -3 | Mega -2 | Mega last | Mega med | E2E -3 | E2E -2 | E2E last | E2E med |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (row for row in rows if row["scope"] == "e2e"):
        lines.append(
            f"| {row['precision'].upper()} | {row['M']} | {row['backend'].upper()} | "
            f"{fmt(row['frontend_third_last_us'])} | {fmt(row['frontend_second_last_us'])} | "
            f"{fmt(row['frontend_last_us'])} | **{fmt(row['frontend_median_us'])}** | "
            f"{fmt(row['mega_third_last_us'])} | {fmt(row['mega_second_last_us'])} | "
            f"{fmt(row['mega_last_us'])} | **{fmt(row['mega_median_us'])}** | "
            f"{fmt(row['target_third_last_us'])} | {fmt(row['target_second_last_us'])} | "
            f"{fmt(row['target_last_us'])} | **{fmt(row['target_median_us'])}** |"
        )

    lines.extend([
        "",
        "## Mega-only timelines",
        "",
        "| Precision | M | Backend | Mega -3 | Mega -2 | Mega last | Mega median |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in (row for row in rows if row["scope"] == "mega"):
        lines.append(
            f"| {row['precision'].upper()} | {row['M']} | {row['backend'].upper()} | "
            f"{fmt(row['mega_third_last_us'])} | {fmt(row['mega_second_last_us'])} | "
            f"{fmt(row['mega_last_us'])} | **{fmt(row['mega_median_us'])}** |"
        )

    text = "\n".join(lines) + "\n"
    (args.directory / "TIMELINE_LAST3.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
