"""Round-5 bit-identity gate for the cc router knobs (DG_FE_CC_LEAN, DG_FE_CC_SELECT=pruned).

The knobs are read once per process (static), so the gate runs twice: `--save ref.pt` under the
baseline env, then `--ref ref.pt` under the knob env. Per seed x rows {1, 2} x quant {mxfp4, qoq}
it records (a) the knob-0 FE outputs topk_idx / topk_weights (ticket path: router + last-arriver
select) and (b) the DG_FE_SELECT_IN_MEGA=1 compact key array (router only), and compares all of
them bit-exactly against the reference file. Single GPU.
Usage: python3 tests/fe5_ident.py --save /tmp/ref.pt ; DG_FE_CC_LEAN=1 python3 tests/fe5_ident.py --ref /tmp/ref.pt
"""
import argparse, os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import deep_gemm  # noqa: E402
from ncu_frontend_tinym import HIDDEN, EXPERTS, make_buffer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--scale", type=float, default=0.05, help="router weight std (0.05 = the bench default)")
    args = ap.parse_args()
    os.environ.setdefault("DG_FE_TINYM_GRID", "auto"); os.environ.setdefault("DG_FE_TINYM_MMA", "cc")
    torch.manual_seed(20260805)
    w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * args.scale).contiguous()
    buf = make_buffer(64)
    out = {}
    for seed in range(args.seeds):
        for rows in (1, 2):
            torch.manual_seed(17000 + seed * 7919 + rows)
            x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
            for quant in ("mxfp4", "qoq"):
                deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant=quant, select_in_mega=0)
                torch.cuda.synchronize()
                idx = buf.topk_idx[:rows].clone(); wts = buf.topk_weights[:rows].clone()
                xq = buf.x[:rows].clone(); xsf = buf.x_sf[:rows].clone()
                deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant=quant, select_in_mega=1)
                torch.cuda.synchronize()
                keys = deep_gemm.fable_frontend_keys(buf).clone()[: rows * EXPERTS]
                out[(seed, rows, quant)] = (idx.cpu(), wts.cpu(), keys.cpu(), xq.cpu(), xsf.cpu())
    env = {k: os.environ.get(k, "") for k in ("DG_FE_CC_LEAN", "DG_FE_CC_SELECT", "DG_FE_SELECT_IN_MEGA", "DG_FE_TINYM_MMA")}
    if args.save:
        torch.save({"env": env, "out": out}, args.save)
        print(f"FE5_IDENT saved {len(out)} cells to {args.save} env={env}")
    if args.ref:
        ref = torch.load(args.ref)
        bad_idx = bad_w = bad_keys = bad_x = 0
        detail = {}
        for k, (idx, wts, keys, xq, xsf) in out.items():
            ridx, rwts, rkeys, rxq, rxsf = ref["out"][k]
            bad_idx += int((idx != ridx).any()); bad_w += int((wts.view(torch.int32) != rwts.view(torch.int32)).any())
            bad_keys += int((keys != rkeys).any())
            nx = int((xq != rxq).sum()); nsf = int((xsf.view(torch.int32) != rxsf.view(torch.int32)).sum())
            if nx or nsf:
                bad_x += 1
                d = detail.setdefault((k[1], k[2]), [0, 0, 0]); d[0] += 1; d[1] += nx; d[2] += nsf
        status = "PASS" if bad_idx == bad_w == bad_keys == bad_x == 0 else "FAIL"
        print(f"FE5_IDENT {status} cells={len(out)} mismatching cells: topk_idx={bad_idx} topk_weights={bad_w} keys={bad_keys} x/x_sf={bad_x} "
              f"env={env} ref_env={ref['env']}")
        for (rows, quant), (cells, nx, nsf) in sorted(detail.items()):
            print(f"  x/x_sf mismatch rows={rows} quant={quant}: cells={cells} x_bytes_diff={nx} x_sf_words_diff={nsf} "
                  f"(x {tuple(out[(0, rows, quant)][3].shape)} {out[(0, rows, quant)][3].dtype}, x_sf {tuple(out[(0, rows, quant)][4].shape)} {out[(0, rows, quant)][4].dtype})")


if __name__ == "__main__":
    main()
