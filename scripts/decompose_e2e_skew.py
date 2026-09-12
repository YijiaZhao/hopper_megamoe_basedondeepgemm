#!/usr/bin/env python3
"""Decompose the customer-method E2E span of an nsys report per rank: FE kernel start / end, the gap to the Mega
kernel, the Mega kernel start / end, graph memcpy nodes between them (DG_PROFILE_FORCE_BALANCED), and the
inter-rank skew (max - min over the 8 devices) of FE start, FE end and Mega start for the last 3 graph replays.
Optionally compares against a Mega-only report (its Mega span and inter-rank Mega-start skew).
Usage: python3 scripts/decompose_e2e_skew.py e2e_report.nsys-rep [--mega-only mega_report.nsys-rep] [--last 3]"""
import argparse, pathlib, sqlite3, statistics, subprocess

FE = ("router_quant_topk_kernel", "router_cc_lean_kernel")


def export(report):
    db = pathlib.Path("/tmp") / f"{report.stem}.decomp.sqlite"
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "--output", str(db), str(report)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    con = sqlite3.connect(db)
    kernels = {}
    for dev, name, start, end in con.execute("""select k.deviceId, s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on s.id = k.shortName where k.graphNodeId is not null order by k.deviceId, k.start"""):
        kernels.setdefault(dev, []).append((name, start, end))
    memcpy = {}
    try:
        for dev, start, end, nbytes in con.execute("""select deviceId, start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY
                where graphNodeId is not null order by deviceId, start"""):
            memcpy.setdefault(dev, []).append((start, end, nbytes))
    except sqlite3.OperationalError:
        pass
    con.close()
    return kernels, memcpy


def replays(kernels, memcpy, dev, last):
    """(fe_start, fe_end, mega_start, mega_end, [memcpy (start, end)]) of the last `last` graph replays on `dev`."""
    ev = kernels[dev]
    megas = [e for e in ev if "mega_moe_h20_fused_impl" in e[0]]
    fes = [e for e in ev if e[0] in FE]
    out = []
    for mega in megas[-last:]:
        fe = max((f for f in fes if f[1] < mega[1]), key=lambda f: f[1], default=None)
        mc = [(s, e) for (s, e, _) in memcpy.get(dev, []) if fe is not None and fe[2] <= s and e <= mega[1]]
        out.append((fe[1] if fe else None, fe[2] if fe else None, mega[1], mega[2], mc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=pathlib.Path)
    ap.add_argument("--mega-only", type=pathlib.Path)
    ap.add_argument("--last", type=int, default=3)
    args = ap.parse_args()
    kernels, memcpy = export(args.report)
    devs = sorted(kernels)
    per_dev = {d: replays(kernels, memcpy, d, args.last) for d in devs}
    print(f"# {args.report.name}: last {args.last} graph replays, times in us relative to the earliest FE start of that replay over the {len(devs)} devices")
    for i in range(args.last):
        rows = {d: per_dev[d][i] for d in devs if len(per_dev[d]) > i}
        t0 = min(r[0] for r in rows.values() if r[0] is not None)
        print(f"\n## replay {i - args.last} | device | FE start | FE end | FE span | gap FE end -> Mega start | memcpy nodes (start-end) | Mega start | Mega end | Mega span | FE start -> Mega end |")
        print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        fe_s, fe_e, mg_s, mg_e = [], [], [], []
        for d, (fs, fe, ms, me, mc) in rows.items():
            fe_s.append(fs); fe_e.append(fe); mg_s.append(ms); mg_e.append(me)
            mcs = " ".join(f"{(s - t0) / 1e3:.1f}-{(e - t0) / 1e3:.1f}" for s, e in mc) or "-"
            print(f"| GPU{d} | {(fs - t0) / 1e3:.1f} | {(fe - t0) / 1e3:.1f} | {(fe - fs) / 1e3:.1f} | {(ms - fe) / 1e3:.1f} | {mcs} | "
                  f"{(ms - t0) / 1e3:.1f} | {(me - t0) / 1e3:.1f} | {(me - ms) / 1e3:.1f} | {(me - fs) / 1e3:.1f} |")
        sk = lambda v: (max(v) - min(v)) / 1e3
        g0 = rows[devs[0]]
        print(f"| skew (max - min over devices) | {sk(fe_s):.1f} | {sk(fe_e):.1f} | | | | {sk(mg_s):.1f} | {sk(mg_e):.1f} | | |")
        print(f"GPU0: Mega span {(g0[3] - g0[2]) / 1e3:.1f} us; GPU0 Mega start is {(g0[2] - min(mg_s)) / 1e3:.1f} us after the earliest and "
              f"{(max(mg_s) - g0[2]) / 1e3:.1f} us before the latest Mega start; latest Mega start - GPU0 Mega start = the time GPU0 spends "
              f"inside its kernel waiting for the slowest rank's first NVLink barrier at most.")
    if args.mega_only:
        mk, mm = export(args.mega_only)
        print(f"\n# Mega-only reference {args.mega_only.name}: last {args.last} replays")
        print("| replay | GPU0 Mega span (us) | Mega-start skew over devices (us) | Mega-end skew (us) |")
        print("|---|---:|---:|---:|")
        for i in range(args.last):
            rows = {}
            for d in sorted(mk):
                megas = [e for e in mk[d] if "mega_moe_h20_fused_impl" in e[0]]
                if len(megas) >= args.last:
                    rows[d] = megas[-args.last + i]
            s = [r[1] for r in rows.values()]; e = [r[2] for r in rows.values()]
            g0 = rows[min(rows)]
            print(f"| {i - args.last} | {(g0[2] - g0[1]) / 1e3:.1f} | {(max(s) - min(s)) / 1e3:.1f} | {(max(e) - min(e)) / 1e3:.1f} |")


if __name__ == "__main__":
    main()
