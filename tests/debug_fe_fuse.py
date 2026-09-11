"""Diagnostics for the fused Fable frontend (DG_FP4_FUSE_FE): determinism and equality.

8 ranks. Per quant: FE + Mega twice (y0a, y0b), fused-FE Mega twice (y1a, y1b), then per rank:
equal(y0a, y0b), equal(y1a, y1b), equal(y0a, y1a), the frontend outputs' equality, the per-token
max |diff| and the differing hidden-column range. Env knobs are the kernel's (read per launch).

  DG_FP4_FUSE_FE=1 torchrun --standalone --nproc_per_node=8 tests/debug_fe_fuse.py --quant mxfp4 --global-tokens 8
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import deep_gemm  # noqa: E402
from profile_four_api_h20 import EXPERTS, HIDDEN, WORLD, make_tp_group, prepare_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=("mxfp4", "qoq"), default="mxfp4")
    ap.add_argument("--global-tokens", type=int, default=8)
    ap.add_argument("--backend", default="fused")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    make_tp_group(rank)
    m = max(1, args.global_tokens // WORLD)
    buffer, launch_moe = prepare_backend(args, rank, m, group)
    try:
        torch.manual_seed(20260805)
        router_weight = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
        torch.manual_seed(17000 + rank * 1000003)
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)

        def clear():
            buffer.x[:m].zero_(); buffer.x_sf[:m].zero_()
            buffer.topk_idx[:m].fill_(-1); buffer.topk_weights[:m].zero_()
            torch.cuda.synchronize(); dist.barrier(group=group)

        def snapshot():
            torch.cuda.synchronize()
            return (buffer.x[:m].clone(), buffer.x_sf[:m].clone(), buffer.topk_idx[:m].clone(), buffer.topk_weights[:m].clone())

        def run(fused):
            clear()
            y = torch.zeros(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            if fused:
                launch_moe(y, frontend=(x, router_weight))
            else:
                deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant=args.quant)
                launch_moe(y)
            torch.cuda.synchronize(); dist.barrier(group=group)
            return y, snapshot()

        runs0 = [run(False) for _ in range(args.reps)]
        runs1 = [run(True) for _ in range(args.reps)]

        def eq(a, b):
            return bool(torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a,
                                    b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b))

        def describe(tag, ya, yb):
            d = (ya.float() - yb.float()).abs()
            per_tok = [f"{v:.4g}" for v in d.amax(dim=1).tolist()]
            cols = (d > 0).any(dim=0).nonzero().flatten()
            crng = f"[{cols.min().item()},{cols.max().item()}] n={cols.numel()}" if cols.numel() else "-"
            return f"{tag} equal={int(eq(ya, yb))} per_token_max={per_tok} diff_cols={crng}"

        lines = [f"rank {rank} quant={args.quant} M={args.global_tokens} rows={m}"]
        lines.append("  topk_idx(FE) " + str(runs0[0][1][2].tolist()))
        lines.append("  fe outputs FE-run0 vs FE-run1: " + " ".join(
            f"{n}={int(eq(a, b))}" for n, a, b in zip(("x", "x_sf", "idx", "w"), runs0[0][1], runs0[1][1])))
        lines.append("  fe outputs FE-run0 vs fused-run0: " + " ".join(
            f"{n}={int(eq(a, b))}" for n, a, b in zip(("x", "x_sf", "idx", "w"), runs0[0][1], runs1[0][1])))
        lines.append("  " + describe("y knob0 run0 vs run1:", runs0[0][0], runs0[1][0]))
        lines.append("  " + describe("y knob1 run0 vs run1:", runs1[0][0], runs1[1][0]))
        lines.append("  " + describe("y knob0 vs knob1     :", runs0[0][0], runs1[0][0]))
        lines.append(f"  y knob0 finite={int(torch.isfinite(runs0[0][0]).all())} knob1 finite={int(torch.isfinite(runs1[0][0]).all())} "
                     f"|y0|max={runs0[0][0].float().abs().max().item():.4g} |y1|max={runs1[0][0].float().abs().max().item():.4g}")
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, "\n".join(lines), group=group)
        if rank == 0:
            print("\n".join(gathered), flush=True)
    finally:
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
