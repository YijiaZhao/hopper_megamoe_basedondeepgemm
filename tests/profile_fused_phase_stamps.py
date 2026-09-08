"""Phase-stamp probe for the SM90 fused MegaMoE kernel (Mega-only scope).

Reuses the exact routing / weights / buffer setup of profile_four_api_h20.py
(so M, expert placement and L2 flush match the customer table), but launches
the fused kernel with the built-in globaltimer ``phase_stamps`` buffer enabled
and prints, for rank 0, where the time inside one kernel goes.

Slots (see sm90_fp4_mega_moe_h20_fused_body.inl):
  0 entry(min) 12 init done  8 expert-offset atomics  9 topk write  10 grid sync
  11 expert bcast  1 after NVLink barrier(routing done)  2 dispatch pull done
  3 first math task(min)  4 last L1 end  5 last L2 end
  6 after combine NVLink barrier  7 combine end

Usage (inside the four_api_build container, clocks locked):
  torchrun --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py \
      --quant mxfp4 --global-tokens 2 --iters 20
"""
import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_four_api_h20 as P  # noqa: E402

INT64_MAX = (1 << 63) - 1
MIN_SLOTS = (0, 3)
REPORT = [
    (12, "init done"),
    (8, "expert-offset atomics"),
    (9, "topk write"),
    (10, "grid sync"),
    (11, "expert bcast"),
    (1, "NVLink barrier / routing done"),
    (2, "dispatch pull done"),
    (3, "first math task start"),
    (4, "last L1 task end"),
    (5, "last L2 task end"),
    (6, "after combine NVLink barrier"),
    (7, "combine end (kernel end)"),
]
# SM0-only accumulators (per launch after reset): 13 = entry->after NVLink barrier#1,
# 16 = time spent inside NVLink barrier#1 (includes cross-rank launch skew).
ACCUM = [(13, "SM0: entry->after barrier1"), (16, "SM0: barrier1 wait incl. skew")]
# K-loop stage probe (SM0 thread0, SM cycles @1830MHz): per-stage ns = cycles / count / 1.83
STAGE = [(17, "L1 stage: exposed k+1 full wait"), (18, "L1 stage: k+1 RF decode+LUT"),
         (19, "L1 stage: exposed wgmma drain"), (22, "L1 stage: head-to-head total")]
SM_GHZ = 1.83


PROBE_EXP = int(os.environ.get("PROBE_EXP", "0"))  # 1 skip decode, 2 skip wgmma, 3 both (timing only)


def reset(stamps):
    stamps.zero_()
    for s in MIN_SLOTS:
        stamps[s] = INT64_MAX
    stamps[24] = PROBE_EXP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["mxfp4", "qoq"], default="mxfp4")
    ap.add_argument("--global-tokens", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-stamps", action="store_true",
                    help="launch without phase_stamps (wall-time only) to measure probe overhead")
    args = ap.parse_args()
    args.backend = "fused"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    group = dist.new_group(list(range(P.WORLD)))

    active_rows = P.local_tokens(args.global_tokens, rank)
    local_rows = max(1, args.global_tokens // P.WORLD)
    local_experts = P.EXPERTS // P.WORLD
    buffer, _ = P.prepare_backend(args, rank, local_rows, group)
    kernel = (deep_gemm_fused_kernel(args.quant))
    weights = prepare_weights(args, rank, local_experts)

    stamps = torch.zeros(32, dtype=torch.int64, device="cuda")
    try:
        torch.manual_seed(17000 + rank * 1000003 + args.global_tokens)
        x = torch.randn(local_rows, P.HIDDEN, device="cuda", dtype=torch.bfloat16)
        if args.quant == "mxfp4":
            xq, xs = P.per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)
        else:
            xq, xs = P.per_token_cast_to_int8(x, gran_k=128)
        buffer.x[:local_rows].copy_(xq)
        buffer.x_sf[:local_rows].copy_(xs)
        buffer.topk_idx[:local_rows].fill_(-1)
        buffer.topk_weights[:local_rows].zero_()
        if active_rows:
            token = torch.arange(active_rows, device="cuda", dtype=torch.int64)[:, None]
            slot = torch.arange(P.TOPK, device="cuda", dtype=torch.int64)[None, :]
            global_token = rank * local_rows + token
            idx = slot * local_experts + ((global_token + slot * 7) % local_experts)
            buffer.topk_idx[:active_rows].copy_(idx)
            buffer.topk_weights[:active_rows].fill_(1.0 / P.TOPK)
        y = torch.empty(local_rows, P.HIDDEN, device="cuda", dtype=torch.bfloat16)

        def launch():
            kernel(y, *weights, buffer, cumulative_local_expert_recv_stats=None,
                   activation_clamp=10.0, phase_stamps=None if args.no_stamps else stamps)

        launch()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        if args.no_graph:
            run = launch
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="relaxed"):
                launch()
            torch.cuda.synchronize()
            run = graph.replay
        dist.barrier(group=group)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rows = []
        wall = []
        for i in range(args.warmup + args.iters):
            reset(stamps)
            P.flush_l2_cache()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            if i >= args.warmup:
                s = stamps.cpu().tolist()
                t0 = s[0]
                row = {slot: (s[slot] - t0) / 1000.0 for slot, _ in REPORT}
                row.update({slot: s[slot] / 1000.0 for slot, _ in ACCUM})
                n_st = max(s[21], 1)
                row.update({slot: s[slot] / n_st / SM_GHZ for slot, _ in STAGE})
                row[21] = s[21]
                row[23] = s[23] / n_st * 100.0   # % of k+1 waits where data was NOT yet ready
                rows.append(row)
                wall.append(start.elapsed_time(end) * 1000.0)
        dist.barrier(group=group)

        if rank == 0 and args.no_stamps:
            print(f"\n=== fused {args.quant} M={args.global_tokens} NO-STAMPS wall (us): "
                  f"median {statistics.median(wall):.2f} min {min(wall):.2f} max {max(wall):.2f} ===")
        if rank == 0 and not args.no_stamps:
            med = {slot: statistics.median(r[slot] for r in rows) for slot, _ in REPORT}
            mn = {slot: min(r[slot] for r in rows) for slot, _ in REPORT}
            mx = {slot: max(r[slot] for r in rows) for slot, _ in REPORT}
            print(f"\n=== fused {args.quant} M={args.global_tokens} rank0 phase stamps PROBE_EXP={PROBE_EXP} "
                  f"(us from kernel entry, median of {args.iters}; graph={not args.no_graph}) ===")
            print(f"{'slot':>4} {'phase':<32} {'median':>9} {'min':>9} {'max':>9} {'delta':>9}")
            prev = 0.0
            for slot, name in REPORT:
                print(f"{slot:>4} {name:<32} {med[slot]:>9.2f} {mn[slot]:>9.2f} {mx[slot]:>9.2f} {med[slot]-prev:>9.2f}")
                prev = med[slot]
            for slot, name in ACCUM:
                v = [r[slot] for r in rows]
                print(f"{slot:>4} {name:<32} {statistics.median(v):>9.2f} {min(v):>9.2f} {max(v):>9.2f}")
            n_st = statistics.median(r[21] for r in rows)
            print(f"--- K-loop stage probe (SM0 thread0), {n_st:.0f} L1 stages/launch, ns per stage ---")
            for slot, name in STAGE:
                v = [r[slot] for r in rows]
                print(f"{slot:>4} {name:<32} {statistics.median(v):>9.1f} {min(v):>9.1f} {max(v):>9.1f}")
            v = [r[23] for r in rows]
            print(f"  23 {'k+1 tile NOT ready at wait (%)':<32} {statistics.median(v):>9.1f} {min(v):>9.1f} {max(v):>9.1f}")
            print(f"CUDA-event wall (us): median {statistics.median(wall):.2f}  "
                  f"min {min(wall):.2f}  max {max(wall):.2f}")
    finally:
        buffer.destroy()
        dist.destroy_process_group()


def deep_gemm_fused_kernel(quant):
    import deep_gemm
    return deep_gemm.mxfp4_mega_moe_fused if quant == "mxfp4" else deep_gemm.qoq_mega_moe_fused


def prepare_weights(args, rank, local_experts):
    import deep_gemm
    torch.manual_seed(20260703 + rank * 1000003)
    w1 = torch.randn(local_experts, 2 * P.INTERMEDIATE, P.HIDDEN,
                     device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(local_experts, P.HIDDEN, P.INTERMEDIATE,
                     device="cuda", dtype=torch.bfloat16) * 0.05
    if args.quant == "mxfp4":
        return deep_gemm.transform_mxfp4_weights_for_mega_moe_fused(
            P.quantize_to_mxfp4_fused(w1), P.quantize_to_mxfp4_fused(w2))
    return deep_gemm.transform_qoq_weights_for_mega_moe_fused(
        P.quantize_to_qoq(w1), P.quantize_to_qoq(w2))


if __name__ == "__main__":
    main()
