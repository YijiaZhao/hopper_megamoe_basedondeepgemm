#!/usr/bin/env python3
"""Summarize H20 NSYS reports for one explicitly selected GPU/device.

The delivery metric is the median of the final three target executions on
``--device-id`` (device 0 by default). Split MegaMoE is measured from the L1
kernel start through the L2 kernel end, including the inter-kernel gap. E2E is
measured from the Fable frontend start through MegaMoE completion.
"""
import argparse
import csv
import json
import pathlib
import re
import sqlite3
import statistics
import subprocess

REPORT_RE = re.compile(
    r"(e2e|mega)_(split|fused)_(mxfp4|qoq)_M(2|8|16)\.nsys-rep"
)
METRIC_KEYS = (
    "frontend_us", "router_us", "quant_us", "l1_us", "l2_us",
    "mega_span_us", "target_span_us",
)


def report_metadata(report):
    match = REPORT_RE.fullmatch(report.name)
    if not match:
        raise ValueError(f"unexpected report name: {report.name}")
    scope, backend, quant, m_text = match.groups()
    return scope, backend, quant, int(m_text)


def export_events(report):
    database = pathlib.Path("/tmp") / f"{report.stem}.summary.sqlite"
    subprocess.run([
        "nsys", "export", "--type", "sqlite", "--force-overwrite=true",
        "--output", str(database), str(report),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    connection = sqlite3.connect(database)
    query = """select k.deviceId,s.value,k.start,k.end
               from CUPTI_ACTIVITY_KIND_KERNEL k
               join StringIds s on s.id=k.shortName
               where k.graphNodeId is not null
               order by k.deviceId,k.start"""
    per_device = {}
    try:
        for device, name, start, end in connection.execute(query):
            per_device.setdefault(device, []).append((name, start, end))
    finally:
        connection.close()
    return per_device


def build_calls(events, scope, backend, quant):
    frontend_calls = [event for event in events if event[0] == "router_quant_topk_kernel"]

    if backend == "fused":
        target = f"sm90_{quant}_mega_moe_h20_fused_impl"
        fused_calls = [event for event in events if target in event[0]]
        if scope == "e2e" and len(frontend_calls) != len(fused_calls):
            raise RuntimeError(
                f"frontend/fused count mismatch: {len(frontend_calls)} != {len(fused_calls)}"
            )
        count = len(fused_calls)
        if scope == "e2e":
            for front, mega in zip(frontend_calls[-3:], fused_calls[-3:]):
                if not (front[1] < mega[1] < mega[2] and front[2] <= mega[2]):
                    raise RuntimeError(f"invalid frontend-to-fused ordering: front={front}, fused={mega}")
            return [
                dict(
                    frontend_us=(front[2] - front[1]) / 1e3,
                    router_us=None,
                    quant_us=None,
                    l1_us=None,
                    l2_us=None,
                    mega_span_us=(mega[2] - mega[1]) / 1e3,
                    target_span_us=(mega[2] - front[1]) / 1e3,
                )
                for front, mega in zip(frontend_calls[-count:], fused_calls[-count:])
            ]
        return [
            dict(
                frontend_us=None,
                router_us=None,
                quant_us=None,
                l1_us=None,
                l2_us=None,
                mega_span_us=(mega[2] - mega[1]) / 1e3,
                target_span_us=(mega[2] - mega[1]) / 1e3,
            )
            for mega in fused_calls
        ]

    l1_calls = [event for event in events if "sm90_mxfp4_mega_moe_l1_impl" in event[0]]
    l2_calls = [event for event in events if "sm90_mxfp4_mega_moe_l2_impl" in event[0]]
    if len(l1_calls) != len(l2_calls):
        raise RuntimeError(f"L1/L2 count mismatch: {len(l1_calls)} != {len(l2_calls)}")
    if scope == "e2e" and len(frontend_calls) != len(l1_calls):
        raise RuntimeError(
            f"frontend/split count mismatch: {len(frontend_calls)} != {len(l1_calls)}"
        )
    count = len(l1_calls)
    for l1, l2 in zip(l1_calls[-3:], l2_calls[-3:]):
        if not (l1[1] < l2[1] < l2[2] and l1[2] <= l2[2]):
            raise RuntimeError(f"invalid L1-to-L2 ordering: L1={l1}, L2={l2}")
    if scope == "e2e":
        for front, l1 in zip(frontend_calls[-3:], l1_calls[-3:]):
            if not (front[1] < l1[1] and front[2] <= l1[2]):
                raise RuntimeError(f"invalid frontend-to-split ordering: front={front}, L1={l1}")
        calls = []
        for front, l1, l2 in zip(frontend_calls[-count:], l1_calls[-count:], l2_calls[-count:]):
            mega_start = min(l1[1], l2[1])
            mega_end = max(l1[2], l2[2])
            calls.append(dict(
                frontend_us=(front[2] - front[1]) / 1e3,
                router_us=None,
                quant_us=None,
                l1_us=(l1[2] - l1[1]) / 1e3,
                l2_us=(l2[2] - l2[1]) / 1e3,
                mega_span_us=(mega_end - mega_start) / 1e3,
                target_span_us=(mega_end - front[1]) / 1e3,
            ))
        return calls

    calls = []
    for l1, l2 in zip(l1_calls[-count:], l2_calls[-count:]):
        mega_start = min(l1[1], l2[1])
        mega_end = max(l1[2], l2[2])
        duration = (mega_end - mega_start) / 1e3
        calls.append(dict(
            frontend_us=None,
            router_us=None,
            quant_us=None,
            l1_us=(l1[2] - l1[1]) / 1e3,
            l2_us=(l2[2] - l2[1]) / 1e3,
            mega_span_us=duration,
            target_span_us=duration,
        ))
    return calls


def extract_final_three(report, device_id=0):
    scope, backend, quant, m_value = report_metadata(report)
    per_device = export_events(report)
    if device_id not in per_device:
        raise RuntimeError(
            f"{report}: device {device_id} is absent; devices={sorted(per_device)}"
        )
    try:
        calls = build_calls(per_device[device_id], scope, backend, quant)
    except RuntimeError as error:
        raise RuntimeError(f"{report}: {error}") from error
    final_three = calls[-3:]
    if len(final_three) != 3:
        raise RuntimeError(
            f"{report}: only {len(final_three)} target calls on device {device_id}"
        )
    metadata = dict(
        scope=scope,
        backend=backend,
        quant=quant,
        M=m_value,
        device_id=device_id,
        report=report.name,
    )
    return metadata, final_three


def summarize(report, device_id=0):
    metadata, final_three = extract_final_three(report, device_id)
    metrics = {
        key: None if final_three[0][key] is None else statistics.median(
            call[key] for call in final_three
        )
        for key in METRIC_KEYS
    }
    return dict(**metadata, **metrics)


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    rows = [summarize(path, args.device_id) for path in sorted(args.directory.glob("*.nsys-rep"))]
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 reports, found {len(rows)}")
    order_scope = {"e2e": 0, "mega": 1}
    order_quant = {"mxfp4": 0, "qoq": 1}
    order_backend = {"fused": 0, "split": 1}
    rows.sort(key=lambda row: (
        order_scope[row["scope"]], order_quant[row["quant"]],
        row["M"], order_backend[row["backend"]],
    ))

    with open(args.directory / "TIMELINE_TABLE.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.directory / "TIMELINE_TABLE.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(f"Device: GPU {args.device_id}; metric: median of final three target executions.\n")
    print("| Scope | Precision | M | Backend | Frontend | L1 | L2 | Mega span | Target span |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['scope'].upper()} | {row['quant'].upper()} | {row['M']} | "
            f"{row['backend'].upper()} | {fmt(row['frontend_us'])} | {fmt(row['l1_us'])} | "
            f"{fmt(row['l2_us'])} | {fmt(row['mega_span_us'])} | "
            f"{fmt(row['target_span_us'])} |"
        )


if __name__ == "__main__":
    main()
