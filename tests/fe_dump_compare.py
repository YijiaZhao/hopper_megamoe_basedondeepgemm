"""FE output layout gate: run the Fable frontend alone under two env configurations and byte-compare every
buffer the fused Mega consumes.

The FE knobs (DG_FE_CC_LEAN, DG_FE_CC_SELECT, DG_FE_TINYM_MMA, ...) are read once per process, so the tool
forks one child process per configuration (`--dump`), then compares the two dump files buffer by buffer.

Per cell (seed x rows {1, 2} x quant {mxfp4, qoq}) and per configuration the child records, after pre-filling the
buffers with a sentinel (x = 0x00, x_sf = 0, topk_idx = -7, topk_weights = -7.0, ticket / key area untouched):
  select_in_mega=0 launch (ticket + top-8 select in the FE):
    x            [4, 3072] bytes   mode 0 (mxfp4): fp8 e4m3 per K128 group; mode 1 (qoq): int8 whole row
    x_sf         [4, 24]   fp32    mode 0: per-K128 amax / 448; mode 1: whole-row amax / 127 replicated in the 24 slots
    topk_idx     [4, 8]    int64
    topk_weights [4, 8]    fp32
    ticket       [256]     bytes   workspace [0, 256): tickets / hand-off counters
  select_in_mega=1 launch (router only, the Mega prologue selects):
    x_sel1, x_sf_sel1                 the same activation buffers written by that launch
    keys         [4096]    bytes   workspace [65792, 65792 + 4096): compact [token][384] u32 keys (token t at + t * 1536)
    ticket_sel1  [256]     bytes
Rows >= rows are padding rows: they are compared as well (the Mega reads up to 4 rows of x at tiny M) but reported
separately, and they do not fail the gate unless --strict-padding.

For every buffer: mismatch count, first mismatching flat index (and row / column), both values (raw bytes and
decoded), and a classification:
  identical           no differing byte
  zeros/uninitialised every differing element of one side is the pre-fill sentinel (that side did not write it)
  whole-row missing   the differing elements cover whole rows, one side still holds the sentinel there
  rounding            numeric values differ by <= 1 ulp (fp8 / int8: |q_a - q_b| <= 1; fp32: rel <= 2^-20)
  layout shift        one side's data equals the other side shifted by a whole number of rows / K128 groups
  other               none of the above
Optionally (--ref-torch) each dump's x / x_sf is also compared against the torch per-token casts the Mega reference
uses (deep_gemm.utils.per_token_cast_to_fp8 / quantization_qoq_fused.per_token_cast_to_int8), which tells which of
the two configurations is the wrong one.

Usage (single GPU):
  python3 tests/fe_dump_compare.py --env-a "DG_FE_CC_LEAN=0" --env-b "DG_FE_CC_LEAN=1" --seeds 64
  python3 tests/fe_dump_compare.py --env-a "DG_FE_CC_LEAN=1" --env-b "DG_FE_CC_LEAN=1 DG_FE_CC_SELECT=pruned"
  (DG_FE_SELECT_IN_MEGA 0 vs 1 is always covered: the sel1 buffers are compared with the sel0 ones inside each dump.)
"""
import argparse
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

HIDDEN, EXPERTS, TOPK, PAD_ROWS = 3072, 384, 8, 4
KEYS_OFF = 256 + 64 * 1024
SENTINEL_IDX, SENTINEL_W = -7, -7.0

BUFFERS = ("x", "x_sf", "topk_idx", "topk_weights", "ticket", "x_sel1", "x_sf_sel1", "keys", "ticket_sel1")


def _parse_env(spec):
    out = {}
    for kv in spec.replace(",", " ").split():
        k, _, v = kv.partition("=")
        out[k] = v
    return out


# ----------------------------------------------------------------------------------------------------------------
# child: run the FE per cell and dump the buffers
def dump(args):
    import types
    import deep_gemm  # noqa: E402  (after the env is set by the parent)
    os.environ.setdefault("DG_FE_TINYM_GRID", "auto")
    os.environ.setdefault("DG_FE_TINYM_MMA", "cc")
    torch.manual_seed(20260805)
    w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * args.scale).contiguous()
    buf = types.SimpleNamespace(
        x=torch.empty(64, HIDDEN, device="cuda", dtype=torch.float8_e4m3fn),
        x_sf=torch.empty(64, HIDDEN // 128, device="cuda", dtype=torch.float32),
        topk_idx=torch.empty(64, TOPK, device="cuda", dtype=torch.int64),
        topk_weights=torch.empty(64, TOPK, device="cuda", dtype=torch.float32))
    ws = deep_gemm.fable_frontend_workspace(buf, EXPERTS, buf.x.device)

    def prefill():
        buf.x.view(torch.uint8).zero_(); buf.x_sf.zero_()
        buf.topk_idx.fill_(SENTINEL_IDX); buf.topk_weights.fill_(SENTINEL_W)

    out = {}
    for seed in range(args.seeds):
        for rows in args.rows:
            torch.manual_seed(17000 + seed * 7919 + rows)
            x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
            for quant in args.quants:
                cell = {}
                prefill()
                deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant=quant, select_in_mega=0)
                torch.cuda.synchronize()
                cell["x"] = buf.x[:PAD_ROWS].view(torch.uint8).clone().cpu()
                cell["x_sf"] = buf.x_sf[:PAD_ROWS].clone().cpu()
                cell["topk_idx"] = buf.topk_idx[:PAD_ROWS].clone().cpu()
                cell["topk_weights"] = buf.topk_weights[:PAD_ROWS].clone().cpu()
                cell["ticket"] = ws[:256].clone().cpu()
                prefill()
                deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant=quant, select_in_mega=1)
                torch.cuda.synchronize()
                cell["x_sel1"] = buf.x[:PAD_ROWS].view(torch.uint8).clone().cpu()
                cell["x_sf_sel1"] = buf.x_sf[:PAD_ROWS].clone().cpu()
                cell["keys"] = ws[KEYS_OFF:KEYS_OFF + 4096].clone().cpu()
                cell["ticket_sel1"] = ws[:256].clone().cpu()
                cell["hidden"] = x.cpu()
                out[(seed, rows, quant)] = cell
    env = {k: os.environ.get(k, "") for k in ("DG_FE_CC_LEAN", "DG_FE_CC_SELECT", "DG_FE_SELECT_IN_MEGA",
                                                 "DG_FE_TINYM_MMA", "DG_FE_TINYM_GRID", "DG_FE_ROUTER_L2_PERSIST")}
    torch.save({"env": env, "out": out, "rows": args.rows, "quants": args.quants}, args.dump)
    print(f"FE_DUMP wrote {len(out)} cells to {args.dump} env={env}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# parent: compare
def _decode(name, quant, raw_a, raw_b, flat):
    """Human-readable values of element `flat` of the two raw tensors."""
    if name.startswith("x_sf") or name == "topk_weights":
        return f"{raw_a.flatten()[flat].item():.9g}", f"{raw_b.flatten()[flat].item():.9g}"
    if name.startswith("x"):
        a, b = int(raw_a.flatten()[flat]), int(raw_b.flatten()[flat])
        if quant == "qoq":
            da, db = (a - 256 if a >= 128 else a), (b - 256 if b >= 128 else b)
            return f"0x{a:02x} (int8 {da})", f"0x{b:02x} (int8 {db})"
        fa = torch.tensor([a], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()
        fb = torch.tensor([b], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()
        return f"0x{a:02x} (e4m3 {fa:g})", f"0x{b:02x} (e4m3 {fb:g})"
    if name == "keys":
        ka = raw_a.view(torch.int32).flatten()[flat // 4].item() & 0xFFFFFFFF
        kb = raw_b.view(torch.int32).flatten()[flat // 4].item() & 0xFFFFFFFF
        return (f"byte 0x{int(raw_a.flatten()[flat]):02x} (u32 key 0x{ka:08x} expert {0xFFFF - (ka & 0xFFFF)})",
                f"byte 0x{int(raw_b.flatten()[flat]):02x} (u32 key 0x{kb:08x} expert {0xFFFF - (kb & 0xFFFF)})")
    return str(raw_a.flatten()[flat].item()), str(raw_b.flatten()[flat].item())


def _numeric(name, quant, raw):
    """Numeric view for the rounding / sentinel tests (fp32 for scales / weights, int for the quantised bytes)."""
    if name.startswith("x_sf") or name == "topk_weights":
        return raw.float()
    if name.startswith("x"):
        return raw.view(torch.int8).int() if quant == "qoq" else raw.view(torch.float8_e4m3fn).float()
    if name == "topk_idx":
        return raw.float()
    return raw.view(torch.uint8).int()


def _sentinel_mask(name, raw):
    if name.startswith("x_sf"):
        return raw == 0
    if name.startswith("x"):
        return raw == 0
    if name == "topk_idx":
        return raw == SENTINEL_IDX
    if name == "topk_weights":
        return raw == SENTINEL_W
    return torch.zeros_like(raw, dtype=torch.bool)


def _shape_of(name, quant):
    if name.startswith("x_sf"):
        return (PAD_ROWS, HIDDEN // 128)
    if name.startswith("x"):
        return (PAD_ROWS, HIDDEN)
    if name in ("topk_idx", "topk_weights"):
        return (PAD_ROWS, TOPK)
    if name == "keys":
        return (2, EXPERTS * 4)      # two token rows of u32 keys, as bytes
    return (1, 256)


def _bytes(name, raw):
    return raw if raw.dtype == torch.uint8 else raw.contiguous().view(torch.uint8)


def classify(name, quant, rows, raw_a, raw_b):
    """Return (mismatch_count, first_flat_index, class, detail) for one buffer of one cell."""
    ba, bb = _bytes(name, raw_a).flatten(), _bytes(name, raw_b).flatten()
    diff_bytes = ba != bb
    n_bytes = int(diff_bytes.sum())
    if n_bytes == 0:
        return 0, -1, "identical", ""
    shape = _shape_of(name, quant)
    # element-wise (not byte-wise) mismatch mask in the buffer's natural element type
    na, nb = _numeric(name, quant, raw_a).reshape(shape), _numeric(name, quant, raw_b).reshape(shape)
    if name in ("ticket", "ticket_sel1", "keys"):
        ea, eb = ba.reshape(shape), bb.reshape(shape)
        elem_diff = ea != eb
    else:
        elem_diff = (na != nb) | (torch.isnan(na) != torch.isnan(nb))
        # fp8 NaN encodings (0x7f / 0xff) compare unequal to themselves: fall back to the byte mask for those
        if name.startswith("x") and not name.startswith("x_sf") and quant == "mxfp4":
            elem_diff = diff_bytes.reshape(shape)
    n = int(elem_diff.sum())
    first = int(elem_diff.flatten().nonzero()[0]) if n else int(diff_bytes.nonzero()[0])
    r, c = divmod(first, shape[1])
    rows_hit = sorted(set(elem_diff.nonzero()[:, 0].tolist()))
    in_padding = all(rr >= rows for rr in rows_hit) if name not in ("ticket", "ticket_sel1", "keys") else False
    detail = f"rows_hit={rows_hit}{' (padding only)' if in_padding else ''}"
    # 1. sentinel / uninitialised on one side
    if name not in ("ticket", "ticket_sel1", "keys"):
        sa, sb = _sentinel_mask(name, raw_a.reshape(shape)), _sentinel_mask(name, raw_b.reshape(shape))
        if bool(sb[elem_diff].all()) or bool(sa[elem_diff].all()):
            side = "B" if bool(sb[elem_diff].all()) else "A"
            full_rows = [rr for rr in rows_hit if bool(elem_diff[rr].all())]
            if full_rows and len(full_rows) == len(rows_hit):
                return n, first, "whole-row missing", f"{detail} side {side} never wrote rows {full_rows}"
            return n, first, "zeros/uninitialised", f"{detail} side {side} holds the pre-fill sentinel at every differing element"
    # 2. layout shift: B row r == A row r' (r' != r), or a K128-group shift within the row
    if name not in ("ticket", "ticket_sel1"):
        for rr in rows_hit:
            for rp in range(shape[0]):
                if rp != rr and bool(torch.equal(_bytes(name, raw_b).reshape(shape[0], -1)[rr], _bytes(name, raw_a).reshape(shape[0], -1)[rp])):
                    return n, first, "layout shift", f"{detail} B row {rr} == A row {rp}"
        if name.startswith("x") and not name.startswith("x_sf"):
            ga, gb = ba.reshape(shape[0], HIDDEN // 128, 128), bb.reshape(shape[0], HIDDEN // 128, 128)
            for s in range(1, HIDDEN // 128):
                if bool(torch.equal(gb[:, s:], ga[:, :-s])) or bool(torch.equal(ga[:, s:], gb[:, :-s])):
                    return n, first, "layout shift", f"{detail} K128 groups shifted by {s}"
        if name.startswith("x_sf"):
            for s in range(1, HIDDEN // 128):
                if bool(torch.equal(nb[:, s:], na[:, :-s])) or bool(torch.equal(na[:, s:], nb[:, :-s])):
                    return n, first, "layout shift", f"{detail} scale slots shifted by {s}"
    # 3. rounding: 1 ulp for the quantised bytes, tiny relative for fp32
    if name not in ("ticket", "ticket_sel1", "keys", "topk_idx"):
        da, db = na[elem_diff], nb[elem_diff]
        if name.startswith("x") and not name.startswith("x_sf"):
            if quant == "qoq":
                close = bool(((da - db).abs() <= 1).all())
            else:
                # 1 ulp of e4m3: compare the byte encodings as sign-magnitude integers
                ia, ib = ba.reshape(shape)[elem_diff].int(), bb.reshape(shape)[elem_diff].int()
                ma, mb = torch.where(ia >= 128, -(ia - 128), ia), torch.where(ib >= 128, -(ib - 128), ib)
                close = bool(((ma - mb).abs() <= 1).all())
        else:
            close = bool(((da - db).abs() <= (da.abs().clamp_min(1e-30) * 2.0 ** -20)).all())
        if close:
            return n, first, "rounding", f"{detail} max |a-b| = {float((da.float() - db.float()).abs().max()):.3g}"
    return n, first, "other", detail


def compare(args, dump_a, dump_b):
    a, b = torch.load(dump_a), torch.load(dump_b)
    print(f"FE_DUMP_COMPARE A env={a['env']}\n                B env={b['env']}")
    cells = sorted(a["out"].keys())
    assert cells == sorted(b["out"].keys()), "dump cell sets differ"
    totals = {}     # buffer -> [mismatching cells, mismatching elements, classes, padding-only cells]
    failing = 0
    axes = {}       # (buffer) -> {axis -> set of values hit}
    for cell in cells:
        seed, rows, quant = cell
        ca, cb = a["out"][cell], b["out"][cell]
        cell_fail = False
        for name in BUFFERS:
            n, first, cls, detail = classify(name, quant, rows, ca[name], cb[name])
            t = totals.setdefault(name, [0, 0, {}, 0])
            if n:
                padding_only = "(padding only)" in detail
                t[0] += 1; t[1] += n; t[2][cls] = t[2].get(cls, 0) + 1; t[3] += int(padding_only)
                ax = axes.setdefault(name, {"seed": set(), "rows": set(), "quant": set()})
                ax["seed"].add(seed); ax["rows"].add(rows); ax["quant"].add(quant)
                if not padding_only or args.strict_padding:
                    cell_fail = True
                if t[0] <= args.max_print:
                    va, vb = _decode(name, quant, ca[name], cb[name], first)
                    shape = _shape_of(name, quant)
                    r, c = divmod(first, shape[1])
                    print(f"  MISMATCH seed={seed} rows={rows} quant={quant} buffer={name}: {n} elements, first flat={first} "
                          f"(row {r}, col {c}): A={va} B={vb} class={cls} {detail}")
        # DG_FE_SELECT_IN_MEGA 0 vs 1 inside each dump: the router-only launch must leave the same x / x_sf
        for name, alt in (("x", "x_sel1"), ("x_sf", "x_sf_sel1")):
            for side, d in (("A", ca), ("B", cb)):
                n, first, cls, detail = classify(name, quant, rows, d[name], d[alt])
                t = totals.setdefault(f"{side}:{name} sel0 vs sel1", [0, 0, {}, 0])
                if n:
                    t[0] += 1; t[1] += n; t[2][cls] = t[2].get(cls, 0) + 1; t[3] += int("(padding only)" in detail)
                    if "(padding only)" not in detail or args.strict_padding:
                        cell_fail = True
                    if t[0] <= args.max_print:
                        print(f"  MISMATCH seed={seed} rows={rows} quant={quant} {side}: {name} select_in_mega=0 vs 1: {n} elements, first flat={first} class={cls} {detail}")
        failing += int(cell_fail)
    print(f"FE_DUMP_COMPARE cells={len(cells)} failing_cells={failing}")
    for name, (nc, ne, classes, npad) in totals.items():
        if nc:
            ax = axes.get(name)
            axis_txt = ""
            if ax:
                spans = {k: len(v) for k, v in ax.items()}
                all_seeds, all_rows, all_quants = a.get("seeds", None), a["rows"], a["quants"]
                axis_txt = (f" hit: seeds {spans['seed']}/{len({c[0] for c in cells})}, rows {sorted(ax['rows'])} of {sorted(all_rows)}, "
                            f"quants {sorted(ax['quant'])} of {all_quants}")
            print(f"  {name}: mismatching cells={nc} (padding-only {npad}) elements={ne} classes={classes}{axis_txt}")
        else:
            print(f"  {name}: identical in all {len(cells)} cells")
    if args.ref_torch:
        ref_check(a, "A"); ref_check(b, "B")
    status = "PASS" if failing == 0 else "FAIL"
    print(f"FE_DUMP_COMPARE {status}")
    return failing == 0


def ref_check(d, side):
    """x / x_sf of a dump vs the torch per-token casts the Mega correctness reference uses (rows < rows only)."""
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.quantization_qoq_fused import per_token_cast_to_int8
    bad = {}
    for (seed, rows, quant), c in d["out"].items():
        x = c["hidden"].cuda()
        if quant == "qoq":
            xq, xs = per_token_cast_to_int8(x, gran_k=128)
        else:
            xq, xs = per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)
        for name, sfname in (("x", "x_sf"), ("x_sel1", "x_sf_sel1")):
            nx = int((c[name][:rows].cuda() != xq.view(torch.uint8)).sum())
            nsf = int((c[sfname][:rows].cuda() != xs.float()).sum())
            if nx or nsf:
                k = (rows, quant, name)
                bad.setdefault(k, [0, 0, 0]); bad[k][0] += 1; bad[k][1] += nx; bad[k][2] += nsf
    if not bad:
        print(f"  REF_TORCH {side}: x / x_sf byte-identical to the torch per-token casts in every cell (rows < rows)")
    for (rows, quant, name), (nc, nx, nsf) in sorted(bad.items()):
        print(f"  REF_TORCH {side}: rows={rows} quant={quant} {name}: {nc} cells differ from the torch cast (x bytes {nx}, x_sf words {nsf})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-a", default="DG_FE_CC_LEAN=0", help="space / comma separated K=V list for configuration A")
    ap.add_argument("--env-b", default="DG_FE_CC_LEAN=1", help="space / comma separated K=V list for configuration B")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--quants", nargs="+", default=["mxfp4", "qoq"])
    ap.add_argument("--scale", type=float, default=0.05, help="router weight std")
    ap.add_argument("--out", default="/tmp/fe_dump", help="dump file prefix")
    ap.add_argument("--max-print", type=int, default=3, help="per buffer: first N mismatching cells printed")
    ap.add_argument("--strict-padding", action="store_true", help="padding rows (>= rows) also fail the gate")
    ap.add_argument("--ref-torch", action="store_true", help="also compare each dump against the torch per-token casts")
    ap.add_argument("--dump", default="", help="(child) write the dump to this path and exit")
    ap.add_argument("--reuse", action="store_true", help="skip the child runs if the dump files exist")
    args = ap.parse_args()
    if args.dump:
        dump(args); return
    paths = []
    for tag, spec in (("a", args.env_a), ("b", args.env_b)):
        path = f"{args.out}_{tag}.pt"
        paths.append(path)
        if args.reuse and os.path.exists(path):
            continue
        env = dict(os.environ); env.update(_parse_env(spec))
        cmd = [sys.executable, os.path.abspath(__file__), "--dump", path, "--seeds", str(args.seeds), "--scale", str(args.scale),
               "--rows", *map(str, args.rows), "--quants", *args.quants]
        print(f"FE_DUMP child {tag}: {spec}", flush=True)
        subprocess.run(cmd, env=env, check=True)
    ok = compare(args, *paths)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
