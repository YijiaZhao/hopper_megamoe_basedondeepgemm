#!/usr/bin/env python3
"""Build SUMMARY.md from the *.raw.csv exports written by tests/ncu_frontend_tinym_all.sh (standalone FE)."""
import csv, glob, os, sys

def load_raw(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return None
    hdr, units, data = rows[0], rows[1], rows[2]
    return {h: (v, u) for h, u, v in zip(hdr, units, data)}

def num(d, key, default=None):
    if d is None or key not in d:
        return default
    try:
        return float(d[key][0].replace(",", ""))
    except ValueError:
        return default

def unit(d, key):
    return d[key][1] if d and key in d else ""

def to_bytes(v, u):
    scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9,
             "Kbytes": 1e3, "Mbytes": 1e6}.get(u, 1)
    return v * scale

def to_us(v, u):
    return v * {"nsecond": 1e-3, "usecond": 1, "msecond": 1e3, "second": 1e6, "ns": 1e-3, "us": 1, "ms": 1e3, "s": 1e6}.get(u, 1e-3)

def stalls(d, n=3):
    pref, suf = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
    items = []
    for k in d or {}:
        if k.startswith(pref) and k.endswith(suf) and "not_issued" not in k:
            v = num(d, k)
            if v is not None:
                items.append((v, k[len(pref):-len(suf)]))
    items.sort(reverse=True)
    tot = sum(v for v, _ in items) or 1.0
    return ", ".join(f"{name} {v:.2f} ({100*v/tot:.0f}%)" for v, name in items[:n])

def fmt(v, f="{:.1f}"):
    return "n/a" if v is None else f.format(v)

def main(out):
    order = [f"fe_tinym_M{m}_{q}" for q in ("mxfp4", "qoq") for m in (2, 4, 8, 16)] + [f"fe_legacy_M8_{q}" for q in ("mxfp4", "qoq")]
    order += sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(out, "fe_fullk*.raw.csv")))
    print("# NCU: Fable frontend kernel (router_quant_topk_kernel), H20, standalone single GPU, --set full, --clock-control none (1830 MHz lock)\n")
    print("Profiled standalone (tests/ncu_frontend_tinym.py, no torchrun/MegaMoE); reports are named by global M but the kernel only sees rank 0 rows "
          "of the E2E layout: M2 -> 1 row, M4 -> 1, M8 -> 1, M16 -> 2 (so the M2/M4/M8 rows are same-input repeats). tinym = DG_FE_TINYM=1 (shipped default, 3-stage), "
          "legacy = DG_FE_TINYM=0 (8-stage), fullk<grid> = DG_FE_TINYM_GRID=<grid> (full-K router CTAs sized to the SM count + 1 merger CTA). Launch #6 after 5 eager warm-ups; ncu flushes caches before each replay pass (cold L2).\n")
    print("| variant | dur us | grid x block | regs | smem KB | waves/SM | DRAM bytes | DRAM % peak | L2 hit % | SM issue-slot % | achieved occ % (warps/SM) | SM thr % | mem thr % | top-3 stalls (warps/issue, share) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in order:
        p = os.path.join(out, tag + ".raw.csv")
        d = load_raw(p) if os.path.exists(p) else None
        if d is None:
            print(f"| {tag} | missing | | | | | | | | | | | | |")
            continue
        dur = to_us(num(d, "gpu__time_duration.sum", 0.0), unit(d, "gpu__time_duration.sum"))
        grid = fmt(num(d, "launch__grid_size"), "{:.0f}"); blk = fmt(num(d, "launch__block_size"), "{:.0f}")
        regs = fmt(num(d, "launch__registers_per_thread"), "{:.0f}")
        smem = (num(d, "launch__shared_mem_per_block_static", 0.0) or 0.0) + (num(d, "launch__shared_mem_per_block_dynamic", 0.0) or 0.0)
        smem_kb = to_bytes(smem, unit(d, "launch__shared_mem_per_block_static")) / 1024
        waves = fmt(num(d, "launch__waves_per_multiprocessor"), "{:.2f}")
        dram_b = to_bytes(num(d, "dram__bytes.sum", 0.0), unit(d, "dram__bytes.sum"))
        dram_pct = fmt(num(d, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"))
        l2hit = fmt(num(d, "lts__t_sector_hit_rate.pct"))
        issue = fmt(num(d, "sm__inst_issued.avg.pct_of_peak_sustained_active"))
        occ = fmt(num(d, "sm__warps_active.avg.pct_of_peak_sustained_active"))
        occ_w = fmt(num(d, "sm__warps_active.avg.per_cycle_active"), "{:.1f}")
        smthr = fmt(num(d, "sm__throughput.avg.pct_of_peak_sustained_elapsed"))
        memthr = fmt(num(d, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"))
        print(f"| {tag} | {dur:.2f} | {grid} x {blk} | {regs} | {smem_kb:.1f} | {waves} | {dram_b/1e6:.2f} MB | {dram_pct} | {l2hit} | {issue} | {occ} ({occ_w}) | {smthr} | {memthr} | {stalls(d)} |")
    print("\nColumns: dur = gpu__time_duration; DRAM % / SM thr % / mem thr % = pct_of_peak_sustained_elapsed; "
          "SM issue-slot % = sm__inst_issued pct_of_peak_sustained_active; occ = sm__warps_active pct (warps per active SM cycle); "
          "stalls = smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio, share of the summed stall ratios.")
    print("\nPer-report files: <tag>.ncu-rep (open in Nsight Compute GUI), <tag>.details.txt / .details.csv (all sections), <tag>.raw.csv (full metric set), <tag>.run.log.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/raid/kimi/results/ncu_fe")
