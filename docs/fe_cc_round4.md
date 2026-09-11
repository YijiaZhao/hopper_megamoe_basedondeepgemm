# Fable cc router, round 4 (2026-09-11, H20-3e 10.6.131.7, branch perf/cc-router)

Configuration under test: `DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc` with the round-3 defaults (relaxed atomic
ticket, compact 384-key array, router weights in the persisting L2 set-aside). Kernel `router_quant_topk_kernel`
in `csrc/fable_frontend.cu`: 77 router CTAs x 640 threads (5 experts x 4 K-split warps) + 1 quant CTA.

The router is a dedicated kernel: it uses no DeepGEMM GEMM kernel (no `deep_gemm::` GEMM call, no
wgmma/TMA pipeline; `grep -n "deep_gemm\|gemm" csrc/fable_frontend.cu` finds only the namespace and the shared
select header). DeepGEMM itself ships no router; frameworks run the gate as a plain torch/cuBLAS GEMM + `torch.topk`.

## 1. Is 3.84 us reproducible?  Only for back-to-back launches.

`tests/fe_repro_cc.py` (one fresh process per run, 100 stamped launches, each after a 256 MB L2 flush; kernel end =
"topk written" stamp of the last arriver relative to the earliest CTA start) reproduces the round-3 chain exactly
when the launches are back-to-back (flush -> launch -> stamp clone, the `fe_standalone_bench.py` protocol):

    fe_standalone_bench rows 1 mxfp4: chunk0 1.28 / logits 1.79 / keys written 2.56 / ticket won 2.82-3.07 /
    merge done 3.58 / topk written 3.84 (median of 5), FE event 18.7 us.

The same kernel measures 5.4-5.9 us as soon as the host leaves a millisecond-scale gap between two launches
(bisect, rows 1 mxfp4, 20 launches each, ends per launch):

| protocol between two stamped launches | kernel end median / min / p90 (us) |
|---|---|
| clone the stamps only (back-to-back) | 3.58 / 3.33 / 4.86 (bimodal: 3.33-3.84 with occasional 4.9-5.9) |
| clone + 5 ms `time.sleep` | 5.63 / 5.12 / 5.89 |
| per-launch D2H reduction of the stamps (`float()` sync) | 5.63 / 5.12 / 5.89 |
| 105 event-timed launches first, then stamped launches | first stamped launch 3.58, all following 5.4-5.9 |

The extra time is in the router CTAs' weight loads (keys written max 3.84 instead of 2.82: a few CTAs straggle) and
the relaxed-ticket last arriver waits for them (merge done 5.12 instead of 3.58). Neither the CUDA events around the
launch nor the weight layout change it. Inside the FE+Mega graph (8 ranks, `bench_frontend_tinym.py`
DG_FE_STAMPS=1, FE after the previous Mega): keys written 3.58, topk written 6.40 (rows 1 and 2, both quants)
-> in the pipeline the FE kernel is ~6.4 us, not 3.84: the persisting set-aside does not keep the 2.36 MB router
matrix resident across the Mega's weight stream, and the ticket -> select tail then waits on cold-weight stragglers.

### Reproducibility table (`tests/run_fe_repro_cc.sh`, GPU 4, 1830 MHz, 30 fresh processes per cell = 3 chain runs x 10, 100 launches each)

| regime | cell | kernel-end median (every process) | min | p90 | FE CUDA event median (us, CPU-launch-bound) |
|---|---|---|---|---|---|
| back-to-back (stamp clone only) | rows 1 mxfp4 | 3.58 (30/30) | 3.33 | 3.84 | 20.5-21.1 |
| back-to-back | rows 2 mxfp4 | 3.84 (30/30) | 3.58 | 3.84-4.10 | 20.4-22.8 |
| back-to-back | rows 1 qoq | 3.58 (30/30) | 3.33 | 3.84 | 20.4-21.3 |
| back-to-back | rows 2 qoq | 3.84 (30/30) | 3.58 | 3.84-4.10 | 20.6-22.8 |
| host gap (per-launch D2H reduction) | rows 1 mxfp4 | 5.63 (30/30) | 3.58 | 5.89-6.14 | 20.4-20.9 |
| host gap | rows 2 mxfp4 | 5.63 (27) / 5.89 (3) | 3.84 | 5.89-6.14 | 20.5-20.7 |
| host gap | rows 1 qoq | 6.14 (30/30) | 3.58 | 6.40-6.66 | 20.4-21.3 |
| host gap | rows 2 qoq | 6.14 (16) / 6.40 (14) | 3.58 | 6.66 | 20.5-20.8 |
| back-to-back, DG_FE_SELECT_IN_MEGA=1 | rows 1 mxfp4 | 2.30 (30/30) | 2.05 | 2.30 | 18.6-19.2 |
| back-to-back, DG_FE_SELECT_IN_MEGA=1 | rows 2 mxfp4 | 2.30 (30/30) | 2.05 | 2.56 | 18.9-19.7 |
| back-to-back, DG_FE_SELECT_IN_MEGA=1 | rows 1 qoq | 2.30 (30/30) | 2.05 | 2.30 | 18.9-19.3 |
| back-to-back, DG_FE_SELECT_IN_MEGA=1 | rows 2 qoq | 2.30 (30/30) | 2.05 | 2.56 | 19.8-20.5 |

Verdict: within one protocol the number is stable to one 256 ns stamp tick across 30 processes (3.58 rows 1 / 3.84
rows 2 back-to-back; the round-3 "3.84 all cells" was rows-1 3.58-3.84 at the tick boundary). Across protocols it is
not: the same kernel is 5.6-6.4 with host gaps and 6.4 inside the FE+Mega graph. FE event time (18.6-21 us) is the
eager CPU launch cost and does not move with the kernel. Same-day, different-time-of-day check: the chain ran three
times between 12:37 and 13:20 UTC with identical medians (also 19:25 CST bisect runs earlier gave the same 5.63 /
3.58 pair).

## 2. NCU (`--set full --clock-control none`), rows 1 mxfp4 + rows 2 qoq

Files: `/Users/kimiz/Downloads/h20_fused_official/ncu/ncu_fe_cc/` (`*.ncu-rep`, `*.details.txt`, `*.details.csv`,
`*.raw.csv`, `KEY_TABLE.md`). Summary (default cache control | `--cache-control none`):

| | rows 1 mxfp4 | rows 2 qoq | rows 1 mxfp4, no cache ctl | rows 2 qoq, no cache ctl |
|---|---|---|---|---|
| Duration (replayed, us) | 7.71 | 8.13 | 6.08 | 6.75 |
| SM active / elapsed | 55 % | 52 % | 45 % | 48 % |
| Issue slots busy | 23.3 % | 24.0 % | 41.3 % | 38.3 % |
| Achieved occupancy (theoretical 31.25 %) | 30.9 % | 30.9 % | 30.8 % | 30.9 % |
| DRAM read | 2.45 MB (6.6 %) | 2.45 MB (6.3 %) | 1.56 MB | 1.66 MB |
| L2 hit, SM reads | 20.6 % | 33.9 % | 63.6 % | 69.8 % |
| Registers / thread, dyn smem | 96, 118.8 KB (1-CTA/SM pad) | 96, 118.8 KB | 96 | 96 |
| Top stalls (warps / issue-active cycle) | no_instruction 9.2, long_scoreboard 2.9, barrier 2.3 | same pattern | | |

`no_instruction` (instruction fetch) is the top stall: the fully unrolled 24-chunk load/FMA body plus the
last-arriver select is a large code footprint that every SM fetches cold after the L2 flush. Task 3d (L2 persist):
`cudaDevAttrMaxAccessPolicyWindowSize` = 128 MB, persisting-L2 max 37.5 MB, set-aside granted 3.75 MB for the
2.25 MB request -> the window covers the whole matrix; but DRAM still reads 1.56-1.66 MB of the 2.36 MB with the
profiler's L2 left alone, and in-graph the chain is +1 us -> residency is not being kept, coverage is not the issue.

## 3. Round 4 squeeze

### DG_FE_SELECT_IN_MEGA=1 (user's idea; knob, default 0)
FE (mma 8): router CTAs store their keys into the compact array and end; no ticket, no last-arriver select, no topk
write. `mxfp4|qoq_mega_moe_fused` pick the key array up from the buffer's frontend cache; in the fused Mega
prologue one idle warp per token (warps 4, 5; warps 0-2 do the smem / m-barrier init) reads the 384 keys
(`ld.global.cg`), selects (`deep_gemm/include/deep_gemm/impls/fable_cc_select.cuh`: 12 x 8 insertion + merge8 +
softmax, the same code the FE last arriver uses) and writes this rank's topk_idx / topk_weights before the
kernel-start `__syncthreads`; the dispatch warps read them with `ld.global.cg` (not `ld.global.nc`) in that mode.

FE kernel end (last keys written): 2.05-2.30 us back-to-back (vs 3.58-3.84), 3.07-3.33 inside the graph (vs 6.40).

8-rank FE+Mega graph CUDA-event time (`tests/bench_frontend_tinym.py`, n=100, host barrier, GPU0 median us).
knob 0 = the unchanged round-3 FE (two independent runs, v1 and v3, show the run-to-run noise of ~2 us);
knob 1 = keys-only FE + select in the Mega prologue (run v3, bit-identical topk to knob 0, see gate).

| cell | knob 0 (run v1) | knob 0 (run v3) | knob 1 | delta vs v1 / vs v3 |
|---|---|---|---|---|
| mxfp4 M=2  | 80.38 | 94.91 (outlier, first run after JIT) | 77.44 | -2.9 / n.a. |
| mxfp4 M=8  | 81.02 | 79.14 | 76.83 | -4.2 / -2.3 |
| mxfp4 M=16 | 100.70 | 100.83 | 97.44 | -3.3 / -3.4 |
| qoq M=2    | 78.50 | 76.64 | 75.49 | -3.0 / -1.2 |
| qoq M=8    | 79.49 | 77.34 | 76.26 | -3.2 / -1.1 |
| qoq M=16   | 97.66 | 111.55 (outlier) | 95.71 | -2.0 / n.a. |

Gate (`tests/test_select_in_mega.py`, 8 ranks, 50 seeds, M=2 and M=16, both quants): 0 topk_idx and 0 topk_weights
mismatches over 400 + 800 rows per cell, y bit-identical (max |dy| 0, cos_min 0.9999998). Mega first-math shift:
the Mega median moves by < 1 us either way (knob 1 mxfp4 M=8 71.7 vs 72.8 / 79.3); the E2E gain is the FE
(6.4 -> 3.3 us in-graph kernel) minus the select the Mega prologue now hides under its smem / barrier init.

Verdict: knob 1 wins in every cell, -1.1 .. -4.2 us, i.e. above the 0.5 us bar in both quants but of the same
order as the run-to-run noise of the knob-0 baseline (2-3 us between v1 and v3). The knob stays default 0 in the
FE wrapper because a standalone FE launch with the knob on produces no topk_idx (every standalone FE test /
bench would have to opt out); the pipeline (FE immediately followed by `*_mega_moe_fused`) should set
DG_FE_SELECT_IN_MEGA=1. Two bugs found on the way, both fixed in 4a2e535: the Python key-array view was 256 B
off (the workspace starts with the ticket area) and the dispatch warps read topk_idx with ld.global.nc, which
is not coherent with the same kernel's prologue writes.

### Other squeeze items
Not pursued this round (see the report): with the select moved into the Mega the ticket / select / write segments
leave the FE entirely, so (a) threshold-pruned select, (b) ticket variants and (c) fused topk write only matter for
the knob-0 path; (d) the L2-persist verification is in section 2.
