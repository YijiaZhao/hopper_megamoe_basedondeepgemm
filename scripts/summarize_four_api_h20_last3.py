#!/usr/bin/env python3
"""Report medians of the final three complete spans for each H20 timeline."""
import argparse
import csv
import json
import pathlib
import statistics

from summarize_four_api_h20_timelines import extract_final_three


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def median_or_none(values):
    return None if values[0] is None else statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for report in sorted(args.directory.glob("*.nsys-rep")):
        metadata, final_three = extract_final_three(report, args.device_id)
        rows.append(dict(
            scope=metadata["scope"],
            precision=metadata["quant"],
            M=metadata["M"],
            backend=metadata["backend"],
            device_id=args.device_id,
            frontend_median_us=median_or_none([
                call["frontend_us"] for call in final_three
            ]),
            mega_median_us=statistics.median(
                call["mega_span_us"] for call in final_three
            ),
            target_median_us=statistics.median(
                call["target_span_us"] for call in final_three
            ),
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

    index = {
        (row["scope"], row["precision"], row["M"], row["backend"]): row
        for row in rows
    }
    lines = [
        f"Device: GPU {args.device_id}; values are medians of the final three complete spans.\n",
        "| Precision | M | FE Fused | FE Split | E2E Mega Fused | E2E Mega Split | E2E Fused | E2E Split | Mega-only Fused | Mega-only Split |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for precision in ("mxfp4", "qoq"):
        for m_value in (2, 8, 16):
            e2e_fused = index[("e2e", precision, m_value, "fused")]
            e2e_split = index[("e2e", precision, m_value, "split")]
            mega_fused = index[("mega", precision, m_value, "fused")]
            mega_split = index[("mega", precision, m_value, "split")]
            lines.append(
                f"| {precision.upper()} | {m_value} | "
                f"{fmt(e2e_fused['frontend_median_us'])} | "
                f"{fmt(e2e_split['frontend_median_us'])} | "
                f"{fmt(e2e_fused['mega_median_us'])} | "
                f"{fmt(e2e_split['mega_median_us'])} | "
                f"**{fmt(e2e_fused['target_median_us'])}** | "
                f"**{fmt(e2e_split['target_median_us'])}** | "
                f"{fmt(mega_fused['mega_median_us'])} | "
                f"{fmt(mega_split['mega_median_us'])} |"
            )

    text = "\n".join(lines) + "\n"
    (args.directory / "TIMELINE_LAST3.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
