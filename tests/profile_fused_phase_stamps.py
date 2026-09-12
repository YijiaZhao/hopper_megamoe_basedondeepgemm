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
import deep_gemm  # noqa: E402

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
    (36, "last L2 task math start"),
    (37, "last L2 last-stage inputs ready"),
    (40, "last L2 epilogue start"),
    (5, "last L2 task end"),
    (6, "after combine NVLink barrier"),
    (7, "combine end (kernel end)"),
]
# SM0-only accumulators (per launch after reset): 13 = entry->after NVLink barrier#1,
# 16 = time spent inside NVLink barrier#1 (includes cross-rank launch skew).
ACCUM = [(13, "SM0: entry->after barrier1"), (16, "SM0: barrier1 wait incl. skew")]
# K-loop stage probe (SM0 thread0, SM cycles @1830MHz): per-stage ns = cycles / count / 1.83
# BM8 MXFP4 (kKBlocksPerStage == 2): one "stage" = two K128 blocks (12 stages per L1
# task); 18 = both decodes of the stage, 19 = wait<1> + wait<0> drains.
# 30 = both promotes of the stage (QoQ: s2 loads + int32->float + scale), 31 = both
# WGMMA issue blocks (kKBlocksPerStage == 2 loop only).
STAGE = [(17, "L1 stage: exposed k+1 full wait"), (18, "L1 stage: RF decode+LUT"),
         (19, "L1 stage: exposed wgmma drain"), (31, "L1 stage: wgmma issue (both)"),
         (30, "L1 stage: promote (both)"), (22, "L1 stage: head-to-head total")]
SM_GHZ = 1.83


PROBE_EXP = int(os.environ.get("PROBE_EXP", "0"))  # 1 skip decode, 2 skip wgmma, 3 both (timing only)
PROBE_DUMP = int(os.environ.get("PROBE_DUMP", "0"))
# 1: rank0 prints the per-CTA task timeline of the last measured launch (kernel task log,
# slots 64 + sm * 16 + ..., enabled by the magic word in slot 47; see the body).
PROBE_TASKLOG = int(os.environ.get("PROBE_TASKLOG", "0"))
TASKLOG_BASE, TASKLOG_PER_CTA, TASKLOG_MAX_SMS, TASKLOG_MAGIC = 64, 16, 160, 0x5441534B


def print_tasklog(s, t0):
    ctas = []
    for sm in range(TASKLOG_MAX_SMS):
        base = TASKLOG_BASE + sm * TASKLOG_PER_CTA
        count = s[base + 15]
        if count <= 0:
            continue
        entry = s[base + 14]
        tasks = []
        for t in range(min(count, 7)):
            w0, w1 = s[base + 2 * t], s[base + 2 * t + 1]
            meta = (w0 >> 32) & 0xffffffff
            start = (entry + (w0 & 0xffffffff) - t0) / 1000.0
            end = (entry + w1 - t0) / 1000.0
            tasks.append(dict(l2=bool(meta >> 31 & 1), pb=(meta >> 24) & 0x7f, nb=(meta >> 16) & 0xff,
                              ks=(meta >> 8) & 0xff, nks=meta & 0xff, start=start, end=end))
        ctas.append((sm, tasks))
    if not ctas:
        print("TASKLOG: empty (kernel built without the task log?)")
        return
    def pct(v, q):
        if not v:
            return float("nan")
        v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
    first = [c[1][0]['start'] for c in ctas]
    l1_end = [max(t['end'] for t in c[1] if not t['l2']) for c in ctas if any(not t['l2'] for t in c[1])]
    l2_start = [t['start'] for c in ctas for t in c[1] if t['l2']]
    l2_end = [t['end'] for c in ctas for t in c[1] if t['l2']]
    l1_dur = [t['end'] - t['start'] for c in ctas for t in c[1] if not t['l2'] and t['nks'] == 1]
    l2_dur = [t['end'] - t['start'] for c in ctas for t in c[1] if t['l2'] and t['nks'] == 1]
    kend = max(l2_end) if l2_end else max(l1_end)
    idle = [sum(max(0.0, c[1][i + 1]['start'] - c[1][i]['end']) for i in range(len(c[1]) - 1)) +
            max(0.0, kend - c[1][-1]['end']) for c in ctas]
    print(f"TASKLOG: {len(ctas)} CTAs, tasks/CTA hist "
          f"{ {n: sum(1 for c in ctas if len(c[1]) == n) for n in sorted(set(len(c[1]) for c in ctas))} }")
    print(f"TASKLOG: first task start p0/p50/p100 {pct(first,0):.1f}/{pct(first,.5):.1f}/{pct(first,1):.1f} | "
          f"L1 end per CTA p0/p50/p90/p100 {pct(l1_end,0):.1f}/{pct(l1_end,.5):.1f}/{pct(l1_end,.9):.1f}/{pct(l1_end,1):.1f} | "
          f"L2 start p0/p50/p90/p100 {pct(l2_start,0):.1f}/{pct(l2_start,.5):.1f}/{pct(l2_start,.9):.1f}/{pct(l2_start,1):.1f} | "
          f"L2 end p50/p100 {pct(l2_end,.5):.1f}/{pct(l2_end,1):.1f}")
    if l1_dur:
        print(f"TASKLOG: full L1 task dur p50/p100 {pct(l1_dur,.5):.2f}/{pct(l1_dur,1):.2f}  "
              f"full L2 task dur p50/p100 {pct(l2_dur,.5):.2f}/{pct(l2_dur,1):.2f}  "
              f"idle per CTA (gaps + wait for kernel-wide last L2 end) p50/p100 {pct(idle,.5):.1f}/{pct(idle,1):.1f} "
              f"sum {sum(idle):.0f} us")
    ctas.sort(key=lambda c: -c[1][-1]['end'])
    for sm, tasks in ctas[:10]:
        chain = " ".join(f"{'L2' if t['l2'] else 'L1'}[b{t['pb']},n{t['nb']}{('/' + str(t['ks']) + 'of' + str(t['nks'])) if t['nks'] > 1 else ''}]"
                         f"{t['start']:.1f}-{t['end']:.1f}" for t in tasks)
        print(f"TASKLOG: sm{sm:3d} {chain}")  # 1: rank0 prints one line per measured iteration


def reset(stamps):
    stamps.zero_()
    for s in MIN_SLOTS:
        stamps[s] = INT64_MAX
    stamps[24] = PROBE_EXP
    if PROBE_TASKLOG:
        stamps[47] = TASKLOG_MAGIC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["mxfp4", "qoq"], default="mxfp4")
    ap.add_argument("--global-tokens", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--fe-routing", action="store_true",
                    help="the standalone FE runs eagerly before every replay (real routing / quant), the graph holds Mega only")
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

    stamps = torch.zeros(TASKLOG_BASE + TASKLOG_MAX_SMS * TASKLOG_PER_CTA, dtype=torch.int64, device="cuda")
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
        router_weight = None
        if args.fe_routing:
            torch.manual_seed(20260805)
            router_weight = (torch.randn(P.EXPERTS, P.HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
            deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant=args.quant)
            torch.cuda.synchronize()

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
        raw_rows = []
        wall = []
        for i in range(args.warmup + args.iters):
            reset(stamps)
            P.flush_l2_cache()
            if args.fe_routing:
                deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant=args.quant)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            if i >= args.warmup:
                s = stamps.cpu().tolist()
                raw_rows.append(list(s))
                t0 = s[0]
                row = {slot: (s[slot] - t0) / 1000.0 for slot, _ in REPORT}
                row.update({slot: s[slot] / 1000.0 for slot, _ in ACCUM})
                n_st = max(s[21], 1)
                row.update({slot: s[slot] / n_st / SM_GHZ for slot, _ in STAGE})
                row[21] = s[21]
                row[23] = s[23] / n_st * 100.0   # % of k+1 waits where data was NOT yet ready
                rows.append(row)
                wall.append(start.elapsed_time(end) * 1000.0)
                if PROBE_DUMP and rank == 0:
                    # per-iteration raw numbers (us): kernel entry->combine end, SM0 barrier#1 wait,
                    # entry->after barrier#1, CUDA-event wall around graph.replay()
                    print(f"PROBE_ITER {i - args.warmup:02d} slot7 {row[7]:.2f} slot16 {row[16]:.2f} "
                          f"slot13 {row[13]:.2f} slot7_minus_16 {row[7] - row[16]:.2f} "
                          f"wall {wall[-1]:.2f}", flush=True)
        dist.barrier(group=group)

        if rank == 0 and args.no_stamps:
            print(f"\n=== fused {args.quant} M={args.global_tokens} NO-STAMPS wall (us): "
                  f"median {statistics.median(wall):.2f} min {min(wall):.2f} max {max(wall):.2f} ===")
        if rank == 0 and not args.no_stamps and PROBE_TASKLOG:
            print_tasklog(raw_rows[-1], raw_rows[-1][0])
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
            def _task_us(sl, cnt):
                return [ (sr[sl] / max(sr[cnt], 1) / SM_GHZ / 1000.0) for sr in raw_rows ]
            l1t = _task_us(25, 27); l2t = _task_us(26, 28)
            gap = [ (sr[29] / max(sr[27] + sr[28] - 1, 1) / SM_GHZ / 1000.0) for sr in raw_rows ]
            print(f"--- per-task probe (SM0 thread0), us per task: L1 {statistics.median(l1t):.2f} "
                  f"({statistics.median(sr[27] for sr in raw_rows):.0f} tasks)  L2 {statistics.median(l2t):.2f} "
                  f"({statistics.median(sr[28] for sr in raw_rows):.0f} tasks)  inter-task gap {statistics.median(gap):.2f} ---")
            # 34/35 = SM0 loader: cycles / count of the L1 arrival-count spin per L1 task
            arr = [ (sr[34] / max(sr[35], 1) / SM_GHZ / 1000.0) for sr in raw_rows ]
            print(f"--- SM0 loader L1 arrival-count wait: {statistics.median(arr):.2f} us per task "
                  f"({statistics.median(sr[35] for sr in raw_rows):.0f} tasks) ---")
            # 36/37 = SM0 math: first-stage full-barrier wait per L1 task (pool arrival + stage fill)
            fst = [ (sr[36] / max(sr[37], 1) / SM_GHZ / 1000.0) for sr in raw_rows ]
            print(f"--- SM0 math L1 first-stage wait: {statistics.median(fst):.2f} us per task ---")
            # 32/33 = K128 blocks (stream-K units) run by SM0 in L1 / L2
            print(f"--- SM0 K-blocks: L1 {statistics.median(sr[32] for sr in raw_rows):.0f}  "
                  f"L2 {statistics.median(sr[33] for sr in raw_rows):.0f} ---")
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
