"""Summarise tests/fe5c_sweep.sh logs into one markdown table (one row per launch x api).

Columns: cell tag, api, routing (from ROUTING), top-8 set agreement / weight max |diff| / x byte diffs (ROUTER_REF:
FE vs torch router), cos_min / cos_mean / max|dy| / norm_ratio (RESULT: y vs the torch MoE reference), status.
Usage: python3 scripts/summarize_fe5c_sweep.py /raid/kimi/results/fe5c/*.log [--md out.md]
"""
import argparse, os, re, sys

RE_RESULT = re.compile(r"RESULT api=(\S+?)(?: seed=\d+)? finite=(\d) max_abs=(\S+) mean_abs=(\S+) cos_min=(\S+) cos_mean=(\S+) norm_ratio=(\S+)")
RE_ROUTER = re.compile(r"ROUTER_REF api=(\S+?)(?: seed=\d+)? tokens=(\d+) top8_set_agree=(\d+)/(\d+) weight_max_abs_diff\(agreeing\)=(\S+)(?: weight_max_abs_diff\(all\)=(\S+))? x_bytes_diff=(\d+)(?: x_rows_with_diff=(\d+) x_max_\|dq\|=(\S+))? x_sf_diff=(\d+)")
RE_ROUTING = re.compile(r"ROUTING api=(\S+) tokens/rank=(\d+) global_tokens=(\d+) frontend=(\S+) select_in_mega=(\d) force_balanced=(\d) reference=(\S+)(?: DG_FE_CC_LEAN=(\S+))?")
RE_SELMEGA = re.compile(r"SELECT_IN_MEGA api=(\S+) tokens=\d+: .*?: (\d+) tokens differ")
RE_SLOT = re.compile(r"SLOT_CHECK api=(\S+)")


def parse(path):
    rows = {}
    txt = open(path, errors="replace").read()
    for m in RE_ROUTING.finditer(txt):
        r = rows.setdefault(m.group(1), {})
        r.update(tokens=int(m.group(2)), M=int(m.group(3)), sel=int(m.group(5)), fb=int(m.group(6)), lean=m.group(8) or "-")
    for m in RE_ROUTER.finditer(txt):     # aggregate over the seeds of one launch
        r = rows.setdefault(m.group(1), {})
        r["agree_n"] = r.get("agree_n", 0) + int(m.group(3)); r["agree_d"] = r.get("agree_d", 0) + int(m.group(4))
        r["agree"] = f"{r['agree_n']}/{r['agree_d']}"
        r["wdiff"] = f"{max(float(r.get('wdiff', 0)), float(m.group(5))):.3g}"
        r["wdiff_all"] = f"{max(float(r.get('wdiff_all', 0) or 0), float(m.group(6) or 0)):.3g}" if m.group(6) else ""
        r["xdiff"] = r.get("xdiff", 0) + int(m.group(7))
        r["xrows"] = str(int(r.get("xrows", 0) or 0) + int(m.group(8) or 0)) if m.group(8) else ""
        r["xdq"] = str(max(int(float(r.get("xdq", 0) or 0)), int(float(m.group(9) or 0)))) if m.group(9) else ""
        r["sfdiff"] = r.get("sfdiff", 0) + int(m.group(10))
    for m in RE_SELMEGA.finditer(txt):
        rows.setdefault(m.group(1), {})["selmega_bad"] = int(m.group(2))
    for m in RE_RESULT.finditer(txt):     # aggregate over the seeds of one launch: worst case per column
        r = rows.setdefault(m.group(1), {})
        r["n_seeds"] = r.get("n_seeds", 0) + 1
        r["finite"] = min(r.get("finite", 1), int(m.group(2)))
        r["max_abs"] = max(r.get("max_abs", 0.0), float(m.group(3)))
        r["mean_abs"] = max(r.get("mean_abs", 0.0), float(m.group(4)))
        r["cos_min"] = min(r.get("cos_min", 1.0), float(m.group(5)))
        r["cos_mean"] = min(r.get("cos_mean", 1.0), float(m.group(6)))
        nr = float(m.group(7)); r["norm_lo"] = min(r.get("norm_lo", 9.0), nr); r["norm_hi"] = max(r.get("norm_hi", 0.0), nr)
        r["norm_ratio"] = nr if abs(nr - 1) >= abs(r.get("norm_ratio", 1.0) - 1) else r["norm_ratio"]
    # slot check: count "all slots cos>=0.999" vs bad
    for api in rows:
        blk = re.search(r"SLOT_CHECK api=" + re.escape(api) + r"\n((?:  rank\d.*\n?)+)", txt)
        if blk:
            lines = blk.group(1).strip().splitlines()
            # "rankN: [local active experts=.., with >1 rows=..]" alone = every slot cos >= 0.999; anything after the ] = a bad slot
            bad = [l for l in lines if l.split("]", 1)[-1].strip() not in ("", "all slots cos>=0.999")]
            rows[api]["slot"] = f"{len(lines) - len(bad)}/{len(lines)} ranks clean"
    failed = "Traceback" in txt or "AssertionError" in txt or "Error" in txt and "RESULT" not in txt
    return rows, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    hdr = ("| cell | api | M (tok/rank) | seeds | lean | select_in_mega | routing | top-8 agree | w max diff (agree / all) | x bytes diff (rows, max dq) | x_sf diff "
           "| cos_min | cos_mean (min) | max abs dy | mean abs dy (max) | norm ratio (worst) | slot check | status |")
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    n_fail = 0
    for path in sorted(args.logs):
        rows, failed = parse(path)
        tag = os.path.basename(path)[:-4]
        if not rows:
            out.append(f"| {tag} | - | | | | | | | | | | | | | | | | **NO RESULT** {'(error)' if failed else ''} |"); n_fail += 1; continue
        for api, r in rows.items():
            has = "cos_min" in r
            ok = has and r.get("finite", 0) == 1 and r["cos_min"] >= 0.99 and 0.97 <= r["norm_lo"] and r["norm_hi"] <= 1.03 and not failed
            status = "PASS" if ok else ("**FAIL**" if has else "**NO RESULT**")
            n_fail += int(not ok)
            routing = "balanced (forced)" if r.get("fb") == 1 else ("synthetic balanced" if "agree" not in r else "normal (FE top-8)")
            out.append("| {tag} | {api} | {M} ({tok}) | {ns} | {lean} | {sel} | {routing} | {agree} | {w} | {x} | {sf} | {cmin} | {cmean} | {mx} | {mean} | {nr} | {slot} | {st} |".format(
                tag=tag, api=api.replace("_mega_moe_fused", ""), M=r.get("M", ""), tok=r.get("tokens", ""), ns=r.get("n_seeds", ""), lean=r.get("lean", ""), sel=r.get("sel", ""),
                routing=routing, agree=r.get("agree", "-"), w=(f"{r['wdiff']} / {r['wdiff_all']}" if "wdiff" in r else "-"),
                x=(f"{r['xdiff']} ({r['xrows']} rows, {r['xdq']})" if "xdiff" in r else "-"), sf=r.get("sfdiff", "-"),
                cmin=(f"{r['cos_min']:.8f}" if has else "-"), cmean=(f"{r['cos_mean']:.8f}" if has else "-"),
                mx=(f"{r['max_abs']:.4g}" if has else "-"), mean=(f"{r['mean_abs']:.3g}" if has else "-"), nr=(f"{r['norm_ratio']:.6f}" if has else "-"),
                slot=r.get("slot", ""), st=status))
    text = "\n".join(out) + f"\n\n{len(args.logs)} logs, {n_fail} failing rows\n"
    print(text)
    if args.md:
        open(args.md, "w").write(text)


if __name__ == "__main__":
    main()
