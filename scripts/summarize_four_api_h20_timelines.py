#!/usr/bin/env python3
"""Summarize the 24 H20 NSYS reports using the agreed target-span metric.

For each GPU: take the final three executions, use their median, then average
those eight per-GPU medians.  Split MegaMoE is measured from L1 start through
L2 end, including the inter-kernel gap.  E2E target span is frontend start
through MegaMoE end; TP ReduceScatter/AllGather and L2 eviction are excluded.
"""
import argparse
import csv
import json
import pathlib
import re
import sqlite3
import statistics
import subprocess


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(report):
    match = re.fullmatch(r"(e2e|mega)_(split|fused)_(mxfp4|qoq)_M(2|8|16)\.nsys-rep", report.name)
    if not match:
        raise ValueError(report.name)
    scope, backend, quant, m_text = match.groups()
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
    for device, name, start, end in connection.execute(query):
        per_device.setdefault(device, []).append((name, start, end))

    rank_metrics = []
    for _, events in sorted(per_device.items()):
        if quant == "mxfp4":
            frontend_calls = [
                dict(start=e[1], end=e[2], router_us=None, quant_us=None)
                for e in events
                if e[0] in ("router_kernel", "router_quant_topk_kernel")
            ]
        else:
            routers = [e for e in events if "sm90_bf16_gemm_impl" in e[0]]
            quantizers = [e for e in events if "qoq_quant_topk_kernel" in e[0]]
            count = min(len(routers), len(quantizers))
            frontend_calls = [
                dict(
                    start=router[1], end=quantizer[2],
                    router_us=(router[2] - router[1]) / 1e3,
                    quant_us=(quantizer[2] - quantizer[1]) / 1e3,
                )
                for router, quantizer in zip(routers[-count:], quantizers[-count:])
            ]

        if backend == "fused":
            target = f"sm90_{quant}_mega_moe_h20_fused_impl"
            fused = [e for e in events if target in e[0]]
            count = min(len(frontend_calls), len(fused)) if scope == "e2e" else len(fused)
            calls = []
            if scope == "e2e":
                for front, mega in zip(frontend_calls[-count:], fused[-count:]):
                    calls.append(dict(
                        frontend_us=(front["end"] - front["start"]) / 1e3,
                        router_us=front["router_us"], quant_us=front["quant_us"],
                        l1_us=None, l2_us=None,
                        mega_span_us=(mega[2] - mega[1]) / 1e3,
                        target_span_us=(mega[2] - front["start"]) / 1e3,
                    ))
            else:
                for mega in fused:
                    duration = (mega[2] - mega[1]) / 1e3
                    calls.append(dict(frontend_us=None, router_us=None, quant_us=None,
                                      l1_us=None, l2_us=None, mega_span_us=duration,
                                      target_span_us=duration))
        else:
            l1s = [e for e in events if "sm90_mxfp4_mega_moe_l1_impl" in e[0]]
            l2s = [e for e in events if "sm90_mxfp4_mega_moe_l2_impl" in e[0]]
            count = min(len(l1s), len(l2s), len(frontend_calls)) if scope == "e2e" else min(len(l1s), len(l2s))
            calls = []
            if scope == "e2e":
                iterator = zip(frontend_calls[-count:], l1s[-count:], l2s[-count:])
                for front, l1, l2 in iterator:
                    mega_start, mega_end = min(l1[1], l2[1]), max(l1[2], l2[2])
                    calls.append(dict(
                        frontend_us=(front["end"] - front["start"]) / 1e3,
                        router_us=front["router_us"], quant_us=front["quant_us"],
                        l1_us=(l1[2] - l1[1]) / 1e3, l2_us=(l2[2] - l2[1]) / 1e3,
                        mega_span_us=(mega_end - mega_start) / 1e3,
                        target_span_us=(mega_end - front["start"]) / 1e3,
                    ))
            else:
                for l1, l2 in zip(l1s[-count:], l2s[-count:]):
                    mega_start, mega_end = min(l1[1], l2[1]), max(l1[2], l2[2])
                    duration = (mega_end - mega_start) / 1e3
                    calls.append(dict(
                        frontend_us=None, router_us=None, quant_us=None,
                        l1_us=(l1[2] - l1[1]) / 1e3, l2_us=(l2[2] - l2[1]) / 1e3,
                        mega_span_us=duration, target_span_us=duration,
                    ))
        final_three = calls[-3:]
        if len(final_three) != 3:
            raise RuntimeError(f"{report}: only {len(final_three)} target calls on a device")
        rank_metrics.append({
            key: None if final_three[0][key] is None else statistics.median(call[key] for call in final_three)
            for key in final_three[0]
        })

    metrics = {
        key: None if rank_metrics[0][key] is None else mean([entry[key] for entry in rank_metrics])
        for key in rank_metrics[0]
    }
    return dict(scope=scope, backend=backend, quant=quant, M=int(m_text), report=report.name, **metrics)


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in sorted(args.directory.glob("*.nsys-rep"))]
    order_scope = {"e2e": 0, "mega": 1}
    order_quant = {"mxfp4": 0, "qoq": 1}
    order_backend = {"fused": 0, "split": 1}
    rows.sort(key=lambda r: (order_scope[r["scope"]], order_quant[r["quant"]], r["M"], order_backend[r["backend"]]))
    with open(args.directory / "TIMELINE_TABLE.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.directory / "TIMELINE_TABLE.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("| Scope | Precision | M | Backend | Frontend | L1 | L2 | Mega span | Target span |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['scope'].upper()} | {row['quant'].upper()} | {row['M']} | {row['backend'].upper()} | "
              f"{fmt(row['frontend_us'])} | {fmt(row['l1_us'])} | {fmt(row['l2_us'])} | "
              f"{fmt(row['mega_span_us'])} | {fmt(row['target_span_us'])} |")


if __name__ == "__main__":
    main()
