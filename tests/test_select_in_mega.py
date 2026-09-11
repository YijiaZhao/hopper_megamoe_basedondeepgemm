"""8-rank gate for DG_FE_SELECT_IN_MEGA (cc router): for --seeds random hidden rows per rank,
(1) FE with the select in the FE (knob 0) -> reference topk_idx / topk_weights (+ reference y = Mega on them),
(2) FE with the select in the Mega prologue (knob 1) + Mega -> the topk_idx / topk_weights the Mega
wrote into the buffer must be bit-identical to (1), and y must match (cos_min, max |dy|).

  DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc /usr/local/bin/torchrun --standalone --nproc_per_node=8 \\
      tests/test_select_in_mega.py --quant mxfp4 --global-tokens 8 --seeds 50
"""
import argparse, os, sys, types
import torch, torch.distributed as dist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import deep_gemm  # noqa: E402
from profile_four_api_h20 import EXPERTS, HIDDEN, WORLD, make_tp_group, prepare_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=("mxfp4", "qoq"), required=True)
    ap.add_argument("--global-tokens", type=int, choices=(2, 4, 8, 16), default=8)
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args(); args.backend = "fused"
    dist.init_process_group("nccl"); rank = dist.get_rank(); torch.cuda.set_device(rank)
    group = dist.group.WORLD; make_tp_group(rank)
    local_rows = max(1, args.global_tokens // WORLD)
    buffer, launch_moe = prepare_backend(args, rank, local_rows, group)
    torch.manual_seed(20260805)
    w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    bad_idx = bad_w = 0; cos_min = 1.0; dmax = 0.0
    for seed in range(args.seeds):
        torch.manual_seed(17000 + rank * 1000003 + seed * 7919)
        x = torch.randn(local_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        y0 = torch.empty_like(x); y1 = torch.empty_like(x)
        deep_gemm.fable_router_quant_topk_frontend(x, w, buffer, quant=args.quant, select_in_mega=0)
        idx0 = buffer.topk_idx[:local_rows].clone(); wt0 = buffer.topk_weights[:local_rows].clone()
        launch_moe(y0); torch.cuda.synchronize(); dist.barrier(group=group)
        buffer.topk_idx[:local_rows].fill_(-7); buffer.topk_weights[:local_rows].fill_(-7.0)
        deep_gemm.fable_router_quant_topk_frontend(x, w, buffer, quant=args.quant, select_in_mega=1)
        assert deep_gemm.fable_frontend_keys(buffer) is not None
        launch_moe(y1); torch.cuda.synchronize(); dist.barrier(group=group)
        idx1 = buffer.topk_idx[:local_rows]; wt1 = buffer.topk_weights[:local_rows]
        bad_idx += int((idx0 != idx1).any(dim=1).sum()); bad_w += int((wt0 != wt1).any(dim=1).sum())
        if rank == 0 and seed < 2:
            print(f"rank0 seed {seed} row0 knob0 idx {idx0[0].tolist()} w {[round(v, 4) for v in wt0[0].tolist()]}\n"
                  f"                   knob1 idx {idx1[0].tolist()} w {[round(v, 4) for v in wt1[0].tolist()]}")
        cos = torch.nn.functional.cosine_similarity(y0.float(), y1.float(), dim=1)
        cos_min = min(cos_min, float(cos.min())); dmax = max(dmax, float((y0.float() - y1.float()).abs().max()))
    t = torch.tensor([bad_idx, bad_w], device="cuda"); dist.all_reduce(t, group=group)
    c = torch.tensor([cos_min], device="cuda"); dist.all_reduce(c, op=dist.ReduceOp.MIN, group=group)
    d = torch.tensor([dmax], device="cuda"); dist.all_reduce(d, op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        n = args.seeds * WORLD * local_rows
        print(f"SELMEGA_GATE quant={args.quant} M={args.global_tokens} rows={n}: topk_idx mismatches {int(t[0])}, "
              f"topk_weights mismatches {int(t[1])}, y cos_min {float(c[0]):.8f}, y max|dy| {float(d[0]):.3e} -> "
              f"{'PASS' if int(t[0]) == 0 and int(t[1]) == 0 and float(c[0]) > 0.9999 else 'FAIL'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
