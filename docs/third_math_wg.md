# Third math warpgroup for the tiny-M L1 loop — register/SMEM feasibility (phase 1)

Branch `perf/third-math-wg`, 2026-09-10. Compile-only study on .7 (`four_api_build`,
CUDA 13.0 nvcc, sm_90a) with `scripts/regbudget_compile.sh` / `scripts/regbudget_matrix.sh`
(offline reproduction of the JIT nvcc line + `-Xptxas -v`) on the cached JIT translation
units of the shipped tiny-M kernels (M16 instantiation: BM8 swapAB, RS decode, 2 K128
blocks per stage, 4 stages, push dispatch, interleaved scheduler):

* `mxfp4` = kernel.sm90_mxfp4_h200_fused_interleaved_push_pdf_lean (kMXFP4, kSplitKL1)
* `qoq`   = kernel.sm90_qoq_h200_fused_interleaved_push_pdf_lean_qis2 (kQoQInlineS2, 2 frag buffers)
* `lean`  = kernel.sm90_mxfp4_h200_fused_interleaved_lean (large-M tier, kNumMaxTokensPerRank 512)

## The real budget is setmaxnreg, not `__launch_bounds__`

The kernel already reconfigures registers per role (`kNumDispatchRegisters` 48,
`kNumNonEpilogueRegisters` 64 with the interleaved scheduler, `kNumEpilogueRegisters`
208; body.inl ~L823). The "168 registers" ptxas prints for the 384-thread build is the
launch allocation; the math WGs run at 208. For 512 threads the launch allocation is 128
and the split must satisfy `64*disp + 64*nonepi + 384*math <= 64512`:

| producers (dispatch / loaders) | math WGs (3) |
|---|---|
| 48 / 64 (today's values) | 144 |
| 48 / 40 or 40 / 40 | 152 |
| 24 / 24 | 160 |

Knobs added (default build unchanged): `-DDG_FP4_MATH_WGS=3` (launch bound
128 + 128*WGS, default epi regs 208 -> 144), `-DDG_FP4_EPI_REGS`, `-DDG_FP4_NONEPI_REGS`,
`-DDG_FP4_DISPATCH_REGS`, `-DDG_FP4_PRINT_SMEM_END` (compile error that names the SMEM totals).

## Results (spill stores / spill loads, bytes; "regs" = launch allocation)

384 threads (2 math WGs), math budget swept, producers at 48/64:

| math regs | mxfp4 | qoq |
|---|---|---|
| 208 (shipped) | 168 regs, 0/0 | 168 regs, 0/0 |
| 200 | 0/0 | 0/0 |
| 192 | 0/0 | 4/4 |
| 184 | 0/0 | 60/68 |
| 176 | 0/0 | 204/236 |
| 168 | 20/64 | 0/0 |
| 160 | 16/40 | 8/8 |
| 152 | 20/44 | 16/16 |
| 144 | 72/92 | 0/0 |
| 136 | 180/200 | 16/40 |
| 128 | 264/384 | 44/64 |
| 208, loaders 40 | 0/0 | 0/0 |
| 208, loaders 32, dispatch 32 | 0/0 | 0/0 |

(The qoq column is non-monotonic between 152 and 192: ptxas allocation noise, not a
pressure cliff — 144 and 168 are both clean.)

512 threads (`-DDG_FP4_MATH_WGS=3`, launch allocation 128):

| config (math / loaders / dispatch) | mxfp4 | qoq | lean (large M) |
|---|---|---|---|
| 128 / 64 / 48 | 204/244 (100+ STL/LDL, many inside the GMMA range) | 28/52 (1 inside the GMMA range) | — |
| 144 / 64 / 48 | 28/52 (20 sites, 4 inside the GMMA range) | 0/0 | 192/200 |
| **152 / 40 / 48** | **4/4** (1 STL in the math-role prologue, 1 LDL after the last GMMA: outside the stage loops) | **0/0** | 140/152 |
| 152 / 40 / 40 | 4/4 (same two sites) | 0/0 | — |
| 160 / 24 / 24 | 0/0 | 8/8 (both in the dispatch role region) | — |

SASS check (`scripts/regbudget_sass.sh`): in every 3-WG build the QGMMA/IGMMA range is
SASS lines ~1100-4300 of the math role (lines ~400-7600); the two 152-budget MXFP4
spill instructions sit at lines 475 and 6415.

## Verdict

* Fitting the math path in **128** registers is not needed and not free: MXFP4 spills
  200+ B at 128 (the live set is frag[2][2][4][4] = 64 regs, 2 accumulator sets = 32,
  prefetched packed words + meta = 20, task result 8, plus the uint2 lut[2][2][4] = 32
  transient LUT gathers of the MXFP4 decode — that last item is why MXFP4 needs ~30 more
  than QoQ). None of the reduction candidates has to be applied.
* With the setmaxnreg split **152 / 40 / 48** the shipped tiny-M code compiles with
  **0 spill (QoQ)** and **4 B outside the stage loops (MXFP4)** — under the 16 B gate. The
  loaders and dispatch warps are clean at 40 (and at 32) with the interleaved scheduler.
  160 / 24 / 24 is also clean for MXFP4 but the QoQ dispatch role spills 8 B at 24.
* The large-M tier (`lean`, kNumMaxTokensPerRank 512) spills 140-190 B at 144-152: a 3-WG
  build must stay tiny-M only (host `DG_FP4_MATH_WGS=3` gated on the BM8 RF swapAB tier,
  like the other tiny-M knobs).
* SMEM: `kInterleavedSMEMEnd` = 184,640 B (stages 173,056 + CD 5,120 + fixed regions),
  headroom 47,808 B under 232,448. The 3-WG partial-accumulator hand-off (3 WGs x 2 row
  groups x 128 threads x 8 fp32 = 24,576 B, or 16,384 B if WG0/WG1 keep their own row
  group in registers) fits with the current 4 stages; no need for 3 stages.

## Phase 2 — implemented (`DG_FP4_MATH_WGS=3`, commits on `perf/third-math-wg`)

Host env `DG_FP4_MATH_WGS` (default 2 = today's kernel, byte-for-byte the same TU: the
`#define DG_FP4_MATH_WGS 2` line is the only difference) selects a 512-thread launch on the
tiny-M tier only (`csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp:580` gate:
MXFP4/QoQ, BM8 swapAB, dense BN256 tiles, 2 K128 blocks per stage, >= 3 stages, interleaved
scheduler, none of half-tile / L2 half-row / tiny-M GEMV / fused L1+L2 / RF-prefetch / QIS2
ilv-rawu8-frags knobs; kernel name suffix `_wg3`, `kNumThreads + 128 * (math_wgs - 2)`).

Design as implemented (body.inl line numbers at 5be26ed):

* `kThreeMathWGs` (L426) + tier static assert (L466). `kNumEpilogueThreads` stays 256: the
  epilogue, combine, dispatch+epilogue and all `kNumEpilogueThreads` barriers keep their
  membership (WG0 + WG1). The third WG (`is_extra_math_wg`, L1981) skips the role-start
  dispatch sync (L2011), runs only the K loop + hand-off of every task and returns after the
  task loop (L5267) — no combine duty.
* K-block rotation loop (L3078-L3300, inside `run_swap_ab_rf`): block g of a task belongs to
  WG g mod 3 (stage g / 2, slot g % 2; ring position / parity from the task-start pipeline
  state, shared state advanced by the whole task at the end). The owner runs both 128-row
  groups of its block as two units: (g, row group 0) -> frag[0] / set 0, (g, row group 1) ->
  frag[1] / set 1 — each unit is exactly the two-WG K-block decode + 8 RS WGMMAs with the
  row base `rf_n_idx = p * 128` (L2062; `load_packed_rf` / promote / fold read it instead of
  `wg_n_idx`). Pattern per block: issue(g,0); decode(g,1); issue(g,1); wait<1>; [next owned
  block: full barrier, decode(g',0)]; wait<0>; promote set 0 -> `final_accum`, set 1 ->
  `final_accum_g1` (L2117); release the stage. Exactly two distinct WGs own a stage's two
  blocks, so `empty_barriers` keep 8 arrivals; the task mailbox release counts all 12 math
  warps (L808). A WG without a block (short stream-K segment) only releases the mailbox.
* QoQ uses the per-block promote here (`kInlineS2 = ... && !kThreeMathWGs`, L2572): the
  whole-task int32 sets of inline s2 are read-modify-write wgmma operands for the whole K
  loop, so ptxas has to pin two sets + two fragment buffers; at 152 registers that spilled
  166 / 248 B (34 sites inside the IGMMA range; packed-word prefetch off, one chain), while
  the per-block pattern compiles clean. One accumulator chain per set (`kAccChains`, L2629):
  the two-chain layout serialised the QoQ wgmmas (C7512) and spilled inside the MXFP4 loop.
* Hand-off (L4495): every WG stores its two row-group partials to `smem_ksplit_reduce`
  ([WG][row group][8 elements][128 threads] floats = 24,576 B, L636), `bar.sync 9, 384`,
  WG p (< 2) rebuilds `final_accum` for its epilogue row group as (WG0 + WG1) + WG2 (fixed
  order; fp32 reassociation for MXFP4 and, now, for the QoQ per-block promote) and the
  two-WG epilogue (split-K / stream-K publish included) runs unchanged; WG2 returns. WAR
  guard: WG2 `bar.sync 10, 384` before overwriting (skipped on its first task), WG0/WG1
  `bar.arrive 10, 384` after reading (L861).
* Register split 152 / 40 / 48 (L883, L888; `DG_FP4_*_REGS` still override).

### ptxas -v (scripts/wg3_compile.sh, CUDA 13.0, shipped tiny-M TUs; 5be26ed)

| TU | WGS=2 | WGS=3 |
|---|---|---|
| mxfp4 (push_pdf_lean) | 168 regs, 0 / 0 B, 32 QGMMA, 4 full drains | 128 launch regs (math 152), 16 B stack, 20 / 44 B spill: 3 cold sites (the 64-bit `t_kernel_entry` / task-log words, STL at kernel entry, LDL at the task-log globaltimer reads) — none in the stage loop; 32 QGMMA, 4 full drains |
| qoq (push_pdf_lean_qis2) | 168 regs, 0 / 0 B, 32 IGMMA, 4 full drains | 128 launch regs (math 152), 0 / 0 B, 32 IGMMA, 4 full drains (no C7512) |

Rejected on the way: two chains + inline s2 (qoq: 0 B but C7512 — every IGMMA followed by
a full drain; mxfp4 32 / 48 B with 5 sites in the loop), one chain + inline s2 with prefetch
(qoq 166 / 248 B), one chain + inline s2 without prefetch (same).

Numerics (tests/test_four_api_correctness.py, 8 ranks, same seed; WGS=3 vs today's WGS=2):

| point | mxfp4 WGS=2 | mxfp4 WGS=3 | qoq WGS=2 (inline s2) | qoq WGS=3 (per-block promote) |
|---|---|---|---|---|
| T=8 / rank (64 global) | cos_min 0.9999882579 | 0.9999882579 (identical) | 0.9999301434 | 0.9999301434 (identical) |
| global 2 (stream-K) | 0.9999975562 | 0.9999975562 (identical) | — | 0.9999385476 |
| T=16 / rank (128 global) | — | (host keeps 2 WGs) | 0.9999272823 (per-block promote also 0.9999272823) | 0.9999272823 |

The first 3-WG build ran MXFP4 with one accumulator chain and read cos_min 0.9999583364
(mean_abs +36 %) at every point: the Hopper FP8 tensor-core accumulator truncates, so 4
dependent K32 steps in one chain lose precision vs 2 x 2 — the reason the two-WG kernel has
two chains (and DeepGEMM promotes per K128). QoQ (int32) is exact with one chain.

### Perf — probe (scripts/run_probe.sh, rank 0 medians, us from kernel entry; two passes)

| quant | M | knob | first math (3) | last L1 (4) | L1 phase (4-3) | kernel end (7) | slot 22 head-to-head |
|---|---|---|---|---|---|---|---|
| mxfp4 | 8 | WGS=2 | 14.6 / 13.6 | 44.1 / 43.4 | **29.6 / 29.8** | 52.9 / 51.9 | 1342 / 1328 ns per 2-block stage |
| mxfp4 | 8 | WGS=3 | 13.1 / 13.2 | 44.7 / 44.9 | **31.6 / 31.6** | 52.9 / 53.0 | 1791 / 1784 ns per WG0 block (= 1.5 stages -> 1194 / 1189 per stage) |
| mxfp4 | 16 | WGS=2 | 10.3 / 10.9 | 56.9 / 57.4 | **46.6 / 46.6** | 72.1 / 73.4 | 1350 / 1351 |
| mxfp4 | 16 | WGS=3 | 13.5 / 11.2 | 61.5 / 59.1 | **48.0 / 47.9** | 74.8 / 72.2 | 1703 / 1694 (-> 1135 / 1129 per stage) |
| qoq | 8 | WGS=2 | 10.6 / 11.1 | 40.1 / 40.4 | **29.5 / 29.3** | 49.1 / 49.2 | 1214 / 1215 |
| qoq | 8 | WGS=3 | 10.7 / 10.4 | 40.5 / 40.5 | **29.7 / 30.1** | 48.2 / 48.9 | 1688 / 1653 (-> 1125 / 1102 per stage) |
| qoq | 16 | WGS=2 | 23.8 / — | 67.7 / — | **43.8 / —** | 83.1 / — | 1248 |
| qoq | 16 | WGS=3 | 11.4 / — | 57.1 / — | **45.7 / —** | — | — |

Per-WG stage-loop probe (mxfp4 M8, ns; WGS=2 per 2-block stage = 2 decode units per WG,
WGS=3 per owned block = also 2 decode units per WG): decode+LUT (18) 777 / 788 -> 1195 /
1197 (x1.52), promote (30) 150 / 148 -> 525 / 515, exposed k+1 wait (17) 293 / 287 -> 667 /
669, wgmma issue (31) 482 / 462 -> 407 / 413, drain (19) 196 / 197 -> 201 / 202.

**Verdict: the third math WG does not speed up the tiny-M L1 loop** (L1 phase +1.8 / +1.3 us
at M8 / M16 MXFP4, +0.4 / +1.9 us QoQ; kernel end within noise). The same per-WG work
(two decode units) takes 1.5x longer once three WGs share the SM: the loop is bound by
the SM's issue / shared-memory bandwidth (per 2-block stage the two WGs move ~180 KB
through LDS: 32 LUT gathers x 8 B and 2 x 16 B packed words per row pair per unit, plus
the SFA / meta reads), not by the per-WG latency chain the microbench modelled. Adding a
WG just shares the same issue slots (3 warps per scheduler instead of 2) and, with the
4-stage ring, exposes more of the k+1 wait (a stage is released only after its two owners
also have their next block, so the loader runs at most ~2 stages ahead). Slot 22 per stage
equivalent improves only 10 % (1328 -> 1189 ns) and that is eaten by the hand-off + the
longer first-block wait. Keep `DG_FP4_MATH_WGS=2` as the default; the knob stays as a
documented negative result. The lever for this loop is fewer LDS bytes per K-block
(LUT-free / packed-LUT decode, wider per-thread rows), not more warps.

### Perf — CUDA-event method (tests/bench_frontend_tinym.py --iters 100, GPU0 median / min us, Mega graph; two passes)

| quant | M | WGS=2 pass 1 / 2 | WGS=3 pass 1 / 2 |
|---|---|---|---|
| mxfp4 | 8 | 72.4 (min 69.0) / 88.0 (min 71.4, bimodal) | 73.5 (71.7) / 72.3 (70.5) |
| mxfp4 | 16 | 110.8 (95.5) / 97.0 (94.9) | 96.9 (94.9) / 96.5 (94.8) |
| qoq | 8 | 69.8 (68.0) / 69.6 (67.6) | 69.0 (66.4) / 68.7 (67.1) |
| qoq | 16 | 107.6 (103.6) / 92.2 (90.1) | 93.1 (90.9) / 93.3 (91.3) |

The pass-1 M16 WGS=2 medians (and the pass-2 mxfp4 M8 one) carry a slow mode the
min-over-iterations does not; on the clean cells the knob is within +-1 us of the two-WG
kernel, as the probe says.

Stress: 200 FE+Mega graph replays at M=8 with WGS=3 (spin timeout on): mxfp4 and qoq exit 0,
Mega median 81.4 / 74.6 us (n=200), no hang, no timeout trap.

NSYS_PLACEHOLDER



## Phase 2 sketch (original, superseded by the implementation above)


Rows must stay m64-aligned, so the third WG takes a share of the K-blocks. Stage s of the
2-K-block ring holds 4 units u = (row group p in {0,1}) x (K-block kb in {0,1}), 8 RS
m64n8k32 each (2 halves x 4 K32). Global unit g = 4s + u goes to WG g mod 3: over 3
stages every WG does 4 units (32 wgmma, 1.33 units/stage vs 2 today). Per-WG RF decode is
unchanged (one unit = today's one K-block for one 128-row group), so `frag` stays 64 regs;
each WG needs the task result for both row groups (+8 regs `final_accum`) and, for MXFP4,
still 2 accumulator sets in flight. Expected register cost vs today: about +8 -> still
inside 152 for QoQ, tight for MXFP4 (verify with the same matrix). At task end every WG
stores its partials to SMEM, `bar.sync` over 384 threads, WG p (p < 2) sums the three
partials of row group p and runs today's epilogue with `wg_n_idx = p * 128` (exact int32
for QoQ inline-s2; fp32 reassociation for MXFP4). Sites that hardwire two WGs:

* body.inl L449-L499 static asserts `kNumEpilogueWarpgroups == 2`; L768-769 barrier init
  counts (`empty_barriers` = kNumEpilogueWarps); `kNumEpilogueThreads` (34 uses: named
  barriers incl. dispatch+epilogue syncs at L1334/L1532/L1977, combine warp maps);
  `epilogue_wg_idx` (18) / `wg_n_idx` (22) row split at L2017; the half-tile K-split
  hand-off at L4199-L4240 (`smem_ksplit_reduce`, the template for the partial reduction);
  the interleaved scheduler `release_task_info` per-WG accounting; L2 tier (WG2 idle or
  the same rotation with `L2_WG_BLOCK_N`); host `sm90_fp4_mega_moe_h20_fused.hpp`
  (`kNumEpilogueThreads` in the launch config, `DG_FP4_MATH_WGS` plumbing, tiny-M gate).
