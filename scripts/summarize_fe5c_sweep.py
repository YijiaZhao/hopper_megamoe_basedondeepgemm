"""Summarise tests/fe5c_sweep.sh logs into one markdown table (one row per launch x api).

Columns: cell tag, api, routing (from ROUTING), top-8 set agreement / weight max |diff| / x byte diffs (ROUTER_REF:
FE vs torch router), cos_min / cos_mean / max|dy| / norm_ratio (RESULT: y vs the torch MoE reference), status.
Usage: python3 scripts/summarize_fe5c_sweep.py /raid/kimi/results/fe5c/*.log [--md out.md]
"""
import argparse, os, re, sys

RE_RESULT = re.compile(r"RESULT api=(\S+) finite=(\d) max_abs=(\S+) mean_abs=(\S+) cos_min=(\S+) cos_mean=(\S+) norm_ratio=(\S+)")
RE_ROUTER = re.compile(r"ROUTER_REF api=(\S+) tokens=(\d+) top8_set_agree=(\d+)/(\d+) weight_max_abs_diff\(agreeing\)=(\S+)(?: weight_max_abs_diff\(all\)=(\S+))? x_bytes_diff=(\d+)(?: x_rows_with_diff=(\d+) x_max_\|dq\|=(\S+))? x_sf_diff=(\d+)")
RE_ROUTING = re.compile(r"ROUTING api=(\S+) tokens/rank=(\d+) global_tokens=(\d+) frontend=(\S+) select_in_mega=(\d) force_balanced=(\d) reference=(\S+) DG_FE_CC_LEAN=(\S+)")
RE_SELMEGA = re.compile(r"SELECT_IN_MEGA api=(\S+) tokens=\d+: .*?: (\d+) tokens differ")
RE_SLOT = re.compile(r"SLOT_CHECK api=(\S+)")


def parse(path):
    rows = {}
    txt = open(path, errors="replace").read()
    for m in RE_ROUTING.finditer(txt):
        r = rows.setdefault(m.group(1), {})
        r.update(tokens=int(m.group(2)), M=int(m.group(3)), sel=int(m.group(5)), fb=int(m.group(6)), lean=m.group(8))
    for m in RE_ROUTER.finditer(txt):
        r = rows.setdefault(m.group(1), {})
        r.update(agree=f"{m.group(3)}/{m.group(4)}", wdiff=m.group(5), wdiff_all=m.group(6) or "", xdiff=int(m.group(7)),
                 xrows=m.group(8) or "", xdq=m.group(9) or "", sfdiff=int(m.group(10)))
    for m in RE_SELMEGA.finditer(txt):
        rows.setdefault(m.group(1), {})["selmega_bad"] = int(m.group(2))
    for m in RE_RESULT.finditer(txt):
        r = rows.setdefault(m.group(1), {})
        r.update(finite=int(m.group(2)), max_abs=float(m.group(3)), mean_abs=float(m.group(4)), cos_min=float(m.group(5)),
                 cos_mean=float(m.group(6)), norm_ratio=float(m.group(7)))
    # slot check: count "all slots cos>=0.999" vs bad
    for api in rows:
        blk = re.search(r"SLOT_CHECK api=" + re.escape(api) + r"\n((?:  rank\d.*\n?)+)", txt)
        if blk:
            lines = blk.group(1).strip().splitlines()
            bad = [l for l in lines if "all slots cos>=0.999" not in l]
            rows[api]["slot"] = f"{len(lines) - len(bad)}/{len(lines)} ranks clean"
    failed = "Traceback" in txt or "AssertionError" in txt or "Error" in txt and "RESULT" not in txt
    return rows, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    hdr = ("| cell | api | M (tok/rank) | lean | select_in_mega | routing | top-8 agree | w max diff (agree / all) | x bytes diff (rows, max dq) | x_sf diff "
           "| cos_min | cos_mean | max abs dy | mean abs dy | norm ratio | slot check | status |")
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    n_fail = 0
    for path in sorted(args.logs):
        rows, failed = parse(path)
        tag = os.path.basename(path)[:-4]
        if not rows:
            out.append(f"| {tag} | - | | | | | | | | | | | | | | | **NO RESULT** {'(error)' if failed else ''} |"); n_fail += 1; continue
        for api, r in rows.items():
            has = "cos_min" in r
            ok = has and r.get("finite", 0) == 1 and r["cos_min"] >= 0.99 and 0.97 <= r["norm_ratio"] <= 1.03 and not failed
            status = "PASS" if ok else ("**FAIL**" if has else "**NO RESULT**")
            n_fail += int(not ok)
            routing = "balanced (forced)" if r.get("fb") == 1 else ("synthetic balanced" if "agree" not in r else "normal (FE top-8)")
            out.append("| {tag} | {api} | {M} ({tok}) | {lean} | {sel} | {routing} | {agree} | {w} | {x} | {sf} | {cmin} | {cmean} | {mx} | {mean} | {nr} | {slot} | {st} |".format(
                tag=tag, api=api.replace("_mega_moe_fused", ""), M=r.get("M", ""), tok=r.get("tokens", ""), lean=r.get("lean", ""), sel=r.get("sel", ""),
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
