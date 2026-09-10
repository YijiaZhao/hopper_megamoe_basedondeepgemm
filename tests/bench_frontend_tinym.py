"""Direct CUDA-event timing of the Fable frontend, the fused MegaMoE and the
FE+Mega graph (the customer E2E number without the NCCL collectives).

Per iteration: L2 flush -> torch.cuda.synchronize() -> dist.barrier(WORLD) ->
event0 -> graph.replay() -> event1. Each of the three graphs (FE only, Mega only,
FE+Mega) is timed separately, ITERS times; rank 0 (GPU0) median/min/p90 in us.

  DG_FE_TINYM=0|1 [DG_FE_STAMPS=1] /usr/local/bin/torchrun --standalone --nproc_per_node=8 \\
      tests/bench_frontend_tinym.py --quant mxfp4 --global-tokens 8 [--iters 100]

DG_FE_STAMPS=1 additionally launches the frontend once (eagerly, after an L2
flush) with per-CTA %globaltimer stamps and prints the phase attribution, and
replays a stamped FE+Mega graph (stamps taken inside the graph, i.e. after the
previous replay's Mega streamed its weights through L2).
Knobs echoed: DG_FE_ROUTER_L2_PERSIST (router weights pinned in L2), DG_FE_PDL.
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
from profile_four_api_h20 import (  # noqa: E402
    EXPERTS, HIDDEN, TP, WORLD, flush_l2_cache, make_tp_group, prepare_backend)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def fmt_stats(name, v):
    return f"{name:>8}: median {pct(v, 0.5):7.2f}  min {min(v):7.2f}  p90 {pct(v, 0.9):7.2f} us  (n={len(v)})"


def attribution(stamps_list, num_router_ctas, m):
    rs, qs = [], []
    for st in stamps_list:
        st = st[: num_router_ctas + m].double()
        t0 = st[:, 0].min()
        st = torch.where(st > 0, (st - t0) / 1e3, torch.zeros_like(st))
        rs.append(st[:num_router_ctas]); qs.append(st[num_router_ctas:])
    r, q = torch.cat(rs), torch.cat(qs)

    def line(tag, col):
        return (f"    {tag:<22} median {col.median():6.2f}  min {col.min():6.2f}  max {col.max():6.2f} us")
    print(f"  stamps over {len(stamps_list)} launches (us rel. earliest CTA start of each launch; "
          f"router CTAs={num_router_ctas}, quant CTAs={m}):")
    print(line("router start", r[:, 0])); print(line("router chunk0 landed", r[:, 1]))
    print(line("router mma done", r[:, 2])); print(line("router ticket bumped", r[:, 3]))
    print(line("quant start", q[:, 0])); print(line("quant done", q[:, 1]))
    print(line("ticket seen (topk go)", q[:, 2]))
    if q[:, 4].max() > 0:
        print(line("  partials loaded", q[:, 4])); print(line("  8 rounds done", q[:, 5]))
    print(line("topk done (kernel end)", q[:, 3]))
    late = int((r[:, 0] > r[:, 3].min()).sum())
    print(f"    router CTAs that started after the first router CTA finished (2nd wave): {late}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=("mxfp4", "qoq"), required=True)
    ap.add_argument("--global-tokens", type=int, choices=(2, 4, 8, 16), required=True)
    ap.add_argument("--backend", choices=("split", "fused"), default="fused")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    make_tp_group(rank)
    local_rows = max(1, args.global_tokens // WORLD)
    tinym = int(os.environ.get("DG_FE_TINYM", "1"))
    stamps = int(os.environ.get("DG_FE_STAMPS", "0"))
    l2_persist = int(os.environ.get("DG_FE_ROUTER_L2_PERSIST", "0"))
    pdl = int(os.environ.get("DG_FE_PDL", "0"))
    buffer, launch_moe = prepare_backend(args, rank, local_rows, group)
    try:
        torch.manual_seed(20260805)
        router_weight = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
        torch.manual_seed(17000 + rank * 1000003)
        x = torch.randn(local_rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)

        def fe(st=0):
            deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant=args.quant,
                                                       tinym=tinym, stamps=st)

        def mega():
            launch_moe(y)

        def both():
            fe(); mega()

        def both_stamped():
            fe(1); mega()

        both(); torch.cuda.synchronize(); dist.barrier(group=group)
        graphs = {}
        bodies = [("FE", fe), ("Mega", mega), ("FE+Mega", both)]
        if stamps:
            bodies.append(("FE+Mega/st", both_stamped))
        for name, body in bodies:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s, capture_error_mode="relaxed"):
                body()
            s.synchronize(); graphs[name] = g
            dist.barrier(group=group)

        # Phase-sequential (all FE iterations, then all Mega, then all FE+Mega): the fused
        # MegaMoE's cross-rank flag protocol is captured per graph, so the two graphs that
        # contain it are never interleaved.
        timed = ("FE", "Mega", "FE+Mega")
        times = {k: [] for k in timed}
        for name in timed:
            g = graphs[name]
            torch.cuda.synchronize(); dist.barrier(group=group)
            for it in range(args.warmup + args.iters):
                flush_l2_cache()
                torch.cuda.synchronize()
                dist.barrier(group=group)
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record()
                torch.cuda.synchronize()
                if it >= args.warmup:
                    times[name].append(e0.elapsed_time(e1) * 1e3)
            dist.barrier(group=group)
        if rank == 0:
            print(f"== frontend direct timing: quant={args.quant} M={args.global_tokens} "
                  f"(rows/rank={local_rows}) backend={args.backend} DG_FE_TINYM={tinym} "
                  f"DG_FE_ROUTER_L2_PERSIST={l2_persist} DG_FE_PDL={pdl} iters={args.iters} GPU0 ==")
            for name in timed:
                print(fmt_stats(name, times[name]))
        if stamps:
            collected = []
            for _ in range(5):
                flush_l2_cache(); torch.cuda.synchronize(); dist.barrier(group=group)
                deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant=args.quant,
                                                           tinym=tinym, stamps=1)
                torch.cuda.synchronize()
                collected.append(deep_gemm.fable_frontend_stamps(buffer, EXPERTS))
            if rank == 0:
                print("  [eager FE after L2 flush]")
                attribution(collected, (EXPERTS // 16) * 4, local_rows)
            collected = []
            g = graphs["FE+Mega/st"]
            for _ in range(5):
                flush_l2_cache(); torch.cuda.synchronize(); dist.barrier(group=group)
                g.replay(); torch.cuda.synchronize()
                collected.append(deep_gemm.fable_frontend_stamps(buffer, EXPERTS))
            if rank == 0:
                print("  [FE inside the FE+Mega graph replay, L2 flush + previous Mega before it]")
                attribution(collected, (EXPERTS // 16) * 4, local_rows)
        dist.barrier(group=group)
    finally:
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
