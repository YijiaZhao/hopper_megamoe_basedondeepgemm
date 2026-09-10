#!/usr/bin/env python3
"""Verify the structure and CUDA Graph contents of the NSYS report matrix.

The expected count is derived from the M values present: 8 reports per M
(2 scopes x 2 backends x 2 quants); the customer method (M=2,8,16) gives 24."""
import argparse
import json
import pathlib
import re
import sqlite3
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    reports = sorted(args.directory.glob("*.nsys-rep"))
    results = []
    m_values = set()
    for report in reports:
        match = re.fullmatch(
            r"(e2e|mega)_(split|fused)_(mxfp4|qoq)_M(\d+)\.nsys-rep",
            report.name)
        if not match:
            results.append({"file": report.name, "ok": False, "reason": "unexpected filename"})
            continue
        scope, backend, quant, m_text = match.groups()
        m_values.add(int(m_text))
        database = pathlib.Path("/tmp") / f"{report.stem}.verify.sqlite"
        subprocess.run([
            "nsys", "export", "--type", "sqlite", "--force-overwrite=true",
            "--output", str(database), str(report),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        connection = sqlite3.connect(database)
        rows = connection.execute(
            """select s.value,count(*),
               sum(case when k.graphNodeId is not null then 1 else 0 end),
               count(distinct k.deviceId)
               from CUPTI_ACTIVITY_KIND_KERNEL k
               join StringIds s on s.id=k.shortName group by s.value""").fetchall()
        kernels = {name: (count, graph_count or 0, devices) for name, count, graph_count, devices in rows}

        def graph_present(fragment):
            return sum(graph_count for name, (_, graph_count, _) in kernels.items()
                       if fragment in name) > 0

        checks = {}
        if scope == "e2e":
            checks["fable_frontend"] = graph_present("router_quant_topk_kernel")
        if backend == "fused":
            checks["mega"] = graph_present(f"sm90_{quant}_mega_moe_h20_fused_impl")
        else:
            checks["l1"] = graph_present("sm90_mxfp4_mega_moe_l1_impl")
            checks["l2"] = graph_present("sm90_mxfp4_mega_moe_l2_impl")
        legacy_frontend_absent = not any(
            "qoq_quant_topk_kernel" in name or "mxfp4_quant_topk_kernel" in name
            for name in kernels
        )
        checks["legacy_frontend_absent"] = legacy_frontend_absent
        nccl = [(name, value) for name, value in kernels.items() if "ncclDevKernel" in name]
        nccl_outside_graph = (
            scope != "e2e" or
            (bool(nccl) and all(graph_count == 0 for _, (_, graph_count, _) in nccl)))
        ok = report.stat().st_size > 500_000 and all(checks.values()) and nccl_outside_graph
        results.append({
            "file": report.name,
            "size": report.stat().st_size,
            "expected_graph": checks,
            "nccl_outside_graph": nccl_outside_graph,
            "ok": ok,
        })
    summary = {
        "count": len(reports),
        "m_values": sorted(m_values),
        "expected_count": 8 * len(m_values),
        "all_ok": (len(reports) > 0 and len(reports) == 8 * len(m_values)
                   and all(item["ok"] for item in results)),
        "results": results,
    }
    (args.directory / "VERIFY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"count": summary["count"], "expected_count": summary["expected_count"],
                      "m_values": summary["m_values"], "all_ok": summary["all_ok"]}))
    if not summary["all_ok"]:
        for result in results:
            if not result["ok"]:
                print("BAD", result)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
