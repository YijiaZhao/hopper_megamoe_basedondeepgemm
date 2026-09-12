# Fable cc router, round 5 (2026-09-12, H20-3e 10.6.131.7 standalone / NCU, 10.6.131.8 gates + customer-method campaigns, branch perf/phase-stamps-probe)

Starting point: round 4 (docs/fe_cc_round4.md). Pipeline configuration `DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc
DG_FE_SELECT_IN_MEGA=1`: the FE kernel is the 77 router CTAs (5 experts x 4 K-part warps, weights straight into
registers, 384 compact keys per token) plus one quant CTA; the fused Mega prologue selects the top-8. Customer
method FE column 3.9-4.2 us; standalone kernel end 2.05-2.30 us; NCU top stall `no_instruction`.

Two levers were asked for: (1) the instruction-fetch stall / code footprint of the router kernel, (2) a
threshold-pruned top-8 select. Both are knobs; (1) is a net win and is now the default, (2) is neutral-to-negative
and stays off. A third item fell out of the A/B: the QoQ two-row quantisation on the spare CTA was the kernel's
critical path at rows 2.

## 1. `DG_FE_CC_LEAN` (default 1): a dedicated entry point for the cc44 router

Diagnosis (SASS of the round-4 build, `cuobjdump -sass`): the generic instantiation
`router_quant_topk_kernel<1, kMode, true, true, false, 0, 44, false>` is 10.6 K (mxfp4) / 12.3 K (qoq) SASS
instructions = 170-192 KB of code, because one template body carries every runtime-selected role (polling merger
with 20 / 32 / 40 slots per lane, slot-key and compact-key last arrivers, redux and insertion selects, quant CTA,
PDL variants). The router CTA's hot path starts 73 KB into the kernel (`@!P3 BRA 0x11c50` right after the
prologue), i.e. after a 256 MB L2 flush -- or after the Mega's weight stream in the pipeline -- every CTA fetches
the entry lines and then the far target cold from DRAM before it can even issue its weight loads, and the 20 warps
per CTA then fetch a ~400-instruction straight-line body. The K "loop" itself is only 3 chunks per lane (768 K per
warp / 8 per 16 B / 32 lanes), so `#pragma unroll` variants have nothing to act on; the footprint is the layout,
not the unroll.

`router_cc_lean_kernel<kMode>` (csrc/fable_frontend.cu) is the same algorithm compiled as its own kernel:
* 2.7 K (mxfp4) / 2.9 K (qoq) SASS instructions; the router CTA's code is the first thing in the kernel; the quant
  CTA and the last-arriver select (knob-0 mode) are `__noinline__` functions out of the hot path;
* weight chunks issued first (the DRAM-latency critical stream), then the L2-hot activations;
* activations converted to fp32 once, right after the loads (in the weight-latency shadow); bf16 -> fp32 as one
  LOP3 (high half) / one shift (low half) per element instead of PRMT + IMAD;
* row 1's conversion + FMA chain only when m == 2 (warp-uniform branch; the generic body ran the zero row);
* the spare CTA quantises the two rows concurrently (320 threads and a named barrier per row) instead of one after
  the other: for QoQ (row amax -> barrier -> quantise) the sequential second row made the spare CTA the kernel's
  critical path at rows 2 (3.46 us span vs 2.88 mxfp4 before this change).
Accumulation order is unchanged (per lane: fma chain over the 8 bf16 of a 16 B chunk in element order, chunks
ascending; xor butterfly 16..1; K-part partials summed ks = 0..3; one bf16 rounding), so the keys, top-8, weights
and x / x_sf are bit-identical (section 4). Selected by `launch<>` for kCC == 44 with compact keys + ticket merge
(the defaults); `DG_FE_CC_LEAN=0` = the generic kernel.

### Standalone A/B (`tests/fe5_ab.sh` / `tests/fe5_chain.sh`, GPU 7, 1830 MHz, one fresh process per cell, 200
back-to-back launches each after a 256 MB L2 flush, nsys `cuda_gpu_kern_sum` of the router kernel = GPU span)

| config | cell | base med / avg / min (us) | lean med / avg / min (us) | delta med | stamp kernel end base -> lean |
|---|---|---|---|---|---|
| pipeline (SELECT_IN_MEGA=1) | mxfp4 rows 1 | 2.94 / 2.97 / 2.78 | 2.50 / 2.53 / 2.40 | -0.44 | 2.05 -> 1.79 |
| pipeline | qoq rows 1 | 2.91 / 2.96 / 2.82 | 2.50 / 2.53 / 2.40 | -0.41 | 2.05 -> 1.79 |
| pipeline | mxfp4 rows 2 | 3.17 / 3.18 / 3.04 | 2.88 / 2.93 / 2.82 | -0.29 | 2.30 -> 2.05 |
| pipeline | qoq rows 2 | 3.46 / 3.49 / 3.20 | 2.91 / 2.96 / 2.82 | -0.55 | 2.30 -> 2.05 |
| knob 0 (ticket + select in FE) | mxfp4 rows 1 | 4.48 / 4.57 / 4.32 | 4.06 / 4.09 / 3.94 | -0.42 | 3.58 -> 3.07 |
| knob 0 | qoq rows 1 | 4.48 / 4.57 / 4.32 | 4.03 / 4.07 / 3.87 | -0.45 | 3.58 -> 3.07 |
| knob 0 | mxfp4 rows 2 | 4.64 / 4.72 / 4.48 | 4.42 / 4.42 / 4.26 | -0.22 | 3.84 -> 3.58 |
| knob 0 | qoq rows 2 | 4.64 / 4.73 / 4.51 | 4.42 / 4.42 / 4.22 | -0.22 | 3.84 -> 3.58 |

(rows 2 lean = build eba23b1 with the concurrent two-row quant; the knob-0 rows-2 lean numbers are from the build
before it, 26b932b.) The nsys span sits ~0.7-0.9 us above the stamp chain's kernel end: launch ramp + the last
CTA's completion, which the customer method also contains.

### NCU (`tests/ncu_fe5.sh`, `--set full --clock-control none`, default cache control = cold code and cold weights;
files and KEY_TABLE.md in `~/Downloads/h20_fused_official/ncu/ncu_fe5/`, build 26b932b)

| pipeline configuration, rows 1 mxfp4 | base | lean |
|---|---|---|
| SM elapsed cycles (max) | 9880 | 7291 |
| warp instructions executed | 534.6 K | 298.1 K |
| stall no_instruction (warps / issue-active cycle) | 6.93 | 1.70 |
| stall long_scoreboard | 2.93 | 6.13 |
| registers / thread | 96 | 64 |
| SASS instructions (kernel) | 11.9 K | 2.7 K |

rows 2 qoq: 9050 -> 8106 cycles, 538.7 K -> 394.6 K instructions, no_instruction 6.34 -> 1.16. The top stall moves
from instruction fetch to `long_scoreboard`, i.e. the weight loads: the kernel is now bounded by the DRAM latency of
the 2.36 MB router weight stream (DRAM bytes read unchanged at 2.41-2.45 MB).

## 2. `DG_FE_CC_SELECT=pruned` (default off): two-level threshold select -- neutral, recorded as negative

`fable_cc::select_pruned` (deep_gemm/include/deep_gemm/impls/fable_cc_select.cuh): every lane's max over its 12
keys; the 32 lane maxes ranked by an all-gather (32 shfl.idx + compares, no serial redux chain); bound = the lane max
of rank 7 (8th largest: eight distinct keys are >= it, so the global top-8 is a subset of the keys >= bound); if no
lane holds a second key >= bound (one ballot) the top-8 IS the ranked lane maxes and lane r fetches the lane max of
rank r (fast path, zero redux rounds); otherwise <= 3 candidates per lane (max, 2nd, 3rd max >= bound) and 8 redux
rounds; >= 4 candidates in one lane (ballot, rare) falls back to insertion + merge8. Same 8 keys in the same order on
every path (keys are unique), hence bit-identical topk_idx / topk_weights (section 4). Wired into the FE last arriver
(knob-0 mode, w_hint bit 1024) and into the Mega prologue (`#define DG_FE_CC_SELECT_PRUNED 1` injected into the JIT
header when the env is set).

Result (table above, knob-0 mode, where the select is on the FE's critical path): rows 1 4.48 -> 4.51 us (base) /
4.06 -> 4.10 (lean); rows 2 4.64 -> 4.99 / 4.42 -> 4.70. NCU: 16084 vs 16198 elapsed cycles, 557.8 K vs 557.9 K
instructions. With random logits the fast path is taken for roughly 45 % of the tokens (the probability that the
top-8 land in 8 distinct lanes of 12 experts each); the other tokens pay the rank all-gather plus the 8 redux rounds
they would have paid anyway, and the single last-arriver warp's select is issue-bound at ~1 instruction / cycle, so
neither the all-gather nor the fallback is cheaper than the 12 x 8 insertion it replaces. In the pipeline
configuration the select runs in the Mega prologue under the smem / barrier init and is not on any critical path
(round 4). Knob kept for the record, default `insert`.

## 3. Customer method (`scripts/capture_four_api_h20_timelines.sh`, nsys, GPU 0, median of the last 3 spans, 1830 MHz)

`tests/fe5_campaign.sh`: one build (eba23b1), one session, 5 passes; per pass the four E2E configurations
(base = `DG_FE_CC_LEAN=0`, lean = 1; routing normal = the FE's real top-8, balanced = `DG_PROFILE_FORCE_BALANCED=1`)
and the Mega-only scope, interleaved, so time-of-day drift hits every cell alike. Reported values are the median over
the 5 passes of each capture's GPU-0 median-of-last-3 span. `DG_PROFILE_FORCE_BALANCED=1` (tests/profile_four_api_h20.py):
the FE runs exactly as in the normal mode; one graph memcpy node (cudaMemcpyAsync, no kernel) then overwrites its
compact key array with keys that make the Mega prologue select the Mega-only scope's balanced assignment (local_tokens()
active rows, idx = slot * 48 + (global_token + slot * 7) % 48, weight 1/8; inactive rows get all-zero keys, which the
prologue maps to topk_idx -1). The memcpy sits between the FE and Mega kernels: inside the E2E span, outside both
kernel spans.

### 3a. Plain customer method (build ff7c4ca/14fdb23 python, kernel eba23b1; 10.6.131.8, 8 x H20-3e at 1830 MHz, 5 passes, 2026-09-12 06:58-08:02 UTC)

Values: median over the 5 passes of each capture's GPU-0 median-of-last-3 span (us). FE = the router kernel span,
E2E Mega = the fused Mega kernel span inside the FE + Mega graph, E2E = FE start -> Mega end on GPU 0, Mega-only = the
Mega-only scope (no FE, balanced assignment; identical for both FE builds). `skew` = inter-rank spread (max - min over
the 8 devices) of the Mega kernel start, median over passes of the per-pass maximum of the last 3 replays.

| Precision | M | routing | FE base | FE lean | E2E Mega base | E2E Mega lean | E2E base | E2E lean | Mega-only | skew base / lean (us) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | normal | 3.8 | 2.5 | 69.2 | 66.2 | **73.7** | **69.0** | 45.0 | 12.9 / 19.3 |
| MXFP4 | 2 | balanced | 4.1 | 2.6 | 49.4 | 60.0 | **54.8** | **64.2** | 45.0 | 43.9 / 62.1 |
| MXFP4 | 4 | normal | 4.1 | 2.5 | 69.5 | 90.2 | **73.4** | **93.0** | 54.3 | 13.8 / 50.8 |
| MXFP4 | 4 | balanced | 4.1 | 2.5 | 50.8 | 63.5 | **56.0** | **67.4** | 54.3 | 13.4 / 20.1 |
| MXFP4 | 8 | normal | 4.0 | 2.5 | 72.2 | 76.4 | **76.3** | **79.3** | 60.5 | 14.9 / 20.7 |
| MXFP4 | 8 | balanced | 4.1 | 2.5 | 76.8 | 64.1 | **82.2** | **68.0** | 60.5 | 24.2 / 16.8 |
| MXFP4 | 16 | normal | 4.0 | 2.8 | 89.4 | 92.2 | **93.5** | **95.4** | 78.7 | 20.9 / 13.7 |
| MXFP4 | 16 | balanced | 4.3 | 2.9 | 84.7 | 82.6 | **90.3** | **86.9** | 78.7 | 27.2 / 24.8 |
| QOQ | 2 | normal | 4.2 | 2.7 | 66.0 | 66.3 | **70.5** | **69.3** | 48.7 | 14.7 / 11.9 |
| QOQ | 2 | balanced | 4.3 | 2.8 | 47.3 | 43.9 | **52.7** | **48.2** | 48.7 | 14.5 / 14.7 |
| QOQ | 4 | normal | 4.3 | 2.7 | 72.0 | 69.2 | **76.6** | **72.2** | 50.8 | 23.3 / 17.6 |
| QOQ | 4 | balanced | 4.3 | 2.8 | 58.2 | 53.1 | **64.1** | **57.2** | 50.8 | 12.8 / 18.6 |
| QOQ | 8 | normal | 4.1 | 2.7 | 68.6 | 71.9 | **73.1** | **74.8** | 56.6 | 17.0 / 15.3 |
| QOQ | 8 | balanced | 4.3 | 2.8 | 66.6 | 56.0 | **72.5** | **60.1** | 56.6 | 20.2 / 23.0 |
| QOQ | 16 | normal | 4.4 | 2.8 | 91.4 | 95.3 | **95.7** | **98.5** | 78.5 | 14.3 / 17.5 |
| QOQ | 16 | balanced | 4.3 | 3.0 | 91.2 | 81.4 | **96.6** | **85.8** | 78.5 | 21.1 / 12.6 |

FE per pass (us), base -> lean, every cell 5/5 passes: normal mxfp4 M2 3.8/4.1/3.8/3.8/3.9 -> 2.4/2.5/2.6/2.5/2.5,
M4 4.2/4.7/4.1/3.8/4.0 -> 2.5 x4/2.6, M8 3.8/4.2/4.4/3.9/4.0 -> 2.5/2.6/2.5/2.4/2.5, M16 4.3/4.0/4.1/3.9/3.9 ->
2.9/2.8/2.8/2.8/2.7; qoq M2 4.2/4.1/4.4/4.4/4.1 -> 2.7/2.7/2.8/2.7/2.7, M4 4.4/4.2/4.2/4.4/4.3 -> 2.7/2.7/2.6/2.7/2.6,
M8 4.1/4.1/4.2/4.1/4.1 -> 2.7 x4/2.6, M16 4.4/4.2/4.6/4.3/4.4 -> 2.8/2.8/2.9/2.9/2.8; balanced within 0.1 of normal.
FE verdict: -1.3 .. -1.6 us in every one of the 32 cells (3.8-4.4 -> 2.5-3.0), no overlap between the base and lean
pass distributions; the in-pipeline gain is larger than the standalone -0.3..-0.55 because in the pipeline the
kernel's code is fetched cold after the Mega weight stream (the no_instruction stall the lean entry removes).

E2E / E2E Mega columns: the plain customer method's E2E spans are dominated by inter-rank launch skew (skew column:
12-62 us per cell), not by kernel work, so base-vs-lean differences of +-5..20 us in those two columns are noise of
that skew (see 3b) -- the Mega-only column and the per-rank decomposition below are the kernel-work references.
Passes with skew <= 20 us leave 0-5 of 5 passes per cell (`tests/fe5_summarize_campaign.py <dir> 20` prints the
filtered medians), too few to be a table; the streamed campaign (3c) is the skew-free measurement.

### 3b. Where do the E2E microseconds go? Per-rank decomposition (`scripts/decompose_e2e_skew.py`, base pass 1, balanced M2, last 3 replays; times relative to the earliest FE start of the replay)

MXFP4 M2 balanced, replay -3 (GPU 0 = the earliest rank):

| device | FE start | FE end | FE span | gap FE end -> Mega start | memcpy node | Mega start | Mega end | Mega span | FE start -> Mega end |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| GPU0 | 0.0 | 4.0 | 4.0 | 1.3 | 4.1-5.2 | 5.3 | 60.8 | 55.5 | 60.8 |
| GPU1 | 5.4 | 9.4 | 4.1 | 1.4 | 9.5-10.7 | 10.8 | 60.7 | 50.0 | 55.4 |
| GPU2 | 4.1 | 8.7 | 4.6 | 1.2 | 8.7-9.8 | 9.8 | 60.6 | 50.7 | 56.5 |
| GPU3 | 11.7 | 16.1 | 4.4 | 1.2 | 16.2-17.2 | 17.3 | 60.5 | 43.2 | 48.8 |
| GPU4 | 14.7 | 19.0 | 4.3 | 1.3 | 19.0-20.3 | 20.3 | 61.7 | 41.4 | 47.1 |
| GPU5 | 11.9 | 16.3 | 4.5 | 1.4 | 16.4-17.6 | 17.7 | 60.7 | 43.0 | 48.9 |
| GPU6 | 5.0 | 9.1 | 4.2 | 1.2 | 9.2-10.2 | 10.3 | 60.7 | 50.4 | 55.7 |
| GPU7 | 6.5 | 10.8 | 4.3 | 1.3 | 10.9-12.1 | 12.2 | 60.4 | 48.3 | 53.9 |
| skew (max - min) | 14.7 | 15.0 | | | | 15.0 | 1.3 | | |

Every rank's FE is 4.0-4.6 us; the FE end -> Mega start gap is 1.2-1.5 us, of which the forced-balanced memcpy node
is 1.1-1.2 us (normal routing: gap 0.3 us, no node); the Mega kernels END within 1.3 us of each other on all 8 ranks
(the first in-kernel NVLink barrier aligns them), so each rank's Mega span = common end - its own start: GPU 0
started 15.0 us before the latest rank (GPU4) and its Mega span is 55.5 = 41.4 (the latest rank's span, i.e. the
kernel work, matching the Mega-only column) + 14.1. The other replays: skew 11.4 -> GPU0 Mega span 50.0 (latest rank
39.1); skew 38.4 -> GPU0 72.3 (latest rank 38.2). QOQ M2 balanced: skews 15.5 / 25.1 / 11.8, GPU0 Mega spans 47.3 /
58.8 / 40.2, latest-rank spans 37.7 / 37.3 / 37.2. So the "extra ~10 us" of the E2E balanced cells over Mega-only is
(a) the inter-rank Mega-start skew absorbed by GPU 0 inside the kernel (10-35 us, varies per replay), (b) the 1.2 us
memcpy override node + 0.1 us, (c) the FE itself (4.0 base / 2.5 lean); nothing else is in the E2E span (no extra
kernels). The Mega-only capture has the same effect: its last-3 Mega-start skews were 7.2 / 20.7 / 7.2 us (mxfp4)
and 7.7 / 1530 / 36.2 us (qoq) with Mega-end skews of ~1 us, i.e. GPU 0 was not the earliest rank there, so the
Mega-only column (39.5-46) is closer to the kernel work than the E2E Mega column.

### 3c. Streamed replays ("streamed 30 iters, last-3 median"; `DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30`, `tests/fe5_campaign_streamed.sh`, lean only, 3 passes + 1 Mega-only pass, 08:05-08:21 UTC, same build / box / session)

Same capture and reduction as 3a (nsys, GPU 0, median of the last 3 replays, then median over passes), but inside the
MEASURE loop the 30 replays of a case are enqueued back-to-back (no per-iteration `torch.cuda.synchronize()` /
`dist.barrier()`, no pre-replay barrier; one synchronize + barrier before and after the loop), so the ranks lock to
each other through the on-stream collectives instead of re-skewing at every host round trip. Do not mix with 3a.

| Precision | M | routing | FE lean | E2E Mega lean | E2E lean | Mega-only (streamed) | Mega-start skew lean (us) |
|---|---:|---|---:|---:|---:|---:|---:|
| MXFP4 | 2 | normal | 2.6 | 60.9 | **63.8** | 38.3 | 2.1 |
| MXFP4 | 2 | balanced | 2.7 | 40.7 | **44.7** | 38.3 | 15.8 |
| MXFP4 | 4 | normal | 2.8 | 60.1 | **63.2** | 46.9 | 1.7 |
| MXFP4 | 4 | balanced | 2.7 | 50.1 | **54.0** | 46.9 | 2.1 |
| MXFP4 | 8 | normal | 2.6 | 65.2 | **68.1** | 56.8 | 2.4 |
| MXFP4 | 8 | balanced | 2.7 | 58.3 | **62.3** | 56.8 | 1.8 |
| MXFP4 | 16 | normal | 2.8 | 78.2 | **81.3** | 74.4 | 1.4 |
| MXFP4 | 16 | balanced | 2.8 | 76.8 | **80.8** | 74.4 | 1.1 |
| QOQ | 2 | normal | 2.6 | 60.4 | **63.5** | 37.4 | 50.0 |
| QOQ | 2 | balanced | 2.8 | 42.0 | **46.1** | 37.4 | 15.4 |
| QOQ | 4 | normal | 2.7 | 60.2 | **63.2** | 46.4 | 6.7 |
| QOQ | 4 | balanced | 2.8 | 50.2 | **54.4** | 46.4 | 2.0 |
| QOQ | 8 | normal | 2.7 | 63.4 | **66.3** | 53.3 | 1.5 |
| QOQ | 8 | balanced | 2.8 | 56.5 | **60.6** | 53.3 | 1.4 |
| QOQ | 16 | normal | 3.0 | 76.6 | **80.0** | 73.0 | 1.6 |
| QOQ | 16 | balanced | 3.0 | 76.0 | **80.5** | 73.0 | 1.5 |

Did the ranks lock? Plain method (3a): Mega-start skew 12-62 us per cell. Streamed: 1.1-2.4 us in 12 of 16 cells
(median over passes of the per-pass max of the last 3 replays); the exceptions are the M2 cells (mxfp4 balanced 15.8,
qoq balanced 15.4, qoq normal 50.0: with one active token per TP half the reduce-scatter / all-gather carry almost no
data and do not pin the ranks as tightly; per-replay decomposition of streamed qoq M2 normal pass 1: skews 22.5 / 1.2
/ 50.0) and qoq M4 normal (6.7). Streamed balanced MXFP4 M2, pass 1, last 3 replays (`decompose_e2e_skew.py`): FE
2.7-2.8 us, FE end -> Mega start 1.2-1.4 (memcpy node 1.2), Mega-start skew 0.9 / 2.8 / 6.5 us, GPU 0 Mega span 40.2 /
41.9 / 40.6 vs the streamed Mega-only 38.5 / 38.3 / 38.2 with 1.8-2.6 us skew. With the skew removed the E2E
balanced cells read FE + 1.3 us memcpy node + ~0.1 + Mega (Mega-only + 1-3 us of residual skew), e.g. MXFP4 M2 44.7 =
2.7 + 1.3 + 40.7; and the normal-routing cells show the real cost of the FE's actual routing versus the balanced
assignment: the Mega kernel is 60-65 us at M2-8 with real (unbalanced) top-8 routing against 40-58 balanced.

### 3d. Reference only: plain method WITH `DG_PROFILE_HOST_BARRIER=1` (`dist.barrier()` before every replay; `tests/fe5_campaign_hb.sh`, lean only, 3 passes + 1 Mega-only pass, 08:21-08:39 UTC)

| Precision | M | routing | FE lean | E2E Mega lean | E2E lean | Mega-only (host barrier) | Mega-start skew lean (us) |
|---|---:|---|---:|---:|---:|---:|---:|
| MXFP4 | 2 | normal / balanced | 2.5 / 2.6 | 67.4 / 48.8 | 70.1 / 52.7 | 67.8 | 13.5 / 12.7 |
| MXFP4 | 4 | normal / balanced | 2.5 / 2.6 | 75.7 / 55.2 | 78.5 / 59.1 | 78.5 | 15.4 / 19.3 |
| MXFP4 | 8 | normal / balanced | 2.5 / 2.5 | 71.9 / 63.4 | 74.7 / 67.3 | 86.2 | 153.4 / 12.2 |
| MXFP4 | 16 | normal / balanced | 2.9 / 2.9 | 87.0 / 90.2 | 90.1 / 94.5 | 104.6 | 14.0 / 18.1 |
| QOQ | 2 | normal / balanced | 2.7 / 2.8 | 63.4 / 49.9 | 66.3 / 54.4 | 38.9 | 35.2 / 16.9 |
| QOQ | 4 | normal / balanced | 2.7 / 2.8 | 72.7 / 54.2 | 75.7 / 58.4 | 49.3 | 20.0 / 18.0 |
| QOQ | 8 | normal / balanced | 2.7 / 2.8 | 67.7 / 66.6 | 70.7 / 70.9 | 59.3 | 44.5 / 23.9 |
| QOQ | 16 | normal / balanced | 2.9 / 3.0 | 90.0 / 85.3 | 93.2 / 89.7 | 76.6 | 15.4 / 13.3 |

The host barrier does not remove the skew (12-45 us, one 153 us outlier; the Mega-only mxfp4 pass was badly skewed
at 68-105 us) -- after the barrier every rank still launches its replay from Python independently -- so this set is
kept only as the record that the streamed method (3c), not the barrier, is what locks the ranks.




## 4. Gates

* `tests/fe5_ident.py` (single GPU, 64 seeds x rows {1, 2} x {mxfp4, qoq} = 256 cells, one process per knob setting
  because the knobs are process-static): for every cell the FE output buffers exactly as the Mega consumes them --
  x [64, 3072] float8_e4m3fn (mxfp4: e4m3 per K128 group; qoq: int8 bytes in the same tensor), x_sf [64, 24] fp32,
  topk_idx [64, 8] int64, topk_weights [64, 8] fp32 (rows 0..3 compared, i.e. including the padding rows), the 256 B
  ticket area and the whole 4 KB compact key area at workspace byte 65792 (token t at + t * 384 * 4) after the
  select-in-Mega launch -- compared byte-for-byte against the baseline process (`DG_FE_CC_LEAN=0`):
  `DG_FE_CC_LEAN=1`: PASS 256/256 (0 differing bytes in any buffer); `DG_FE_CC_LEAN=1 DG_FE_CC_SELECT=pruned`: PASS;
  `DG_FE_CC_SELECT=pruned`: PASS. (The first run reported 128 failing x cells; they were the qoq cells of the
  harness comparing the int8 x bytes as float8_e4m3fn values, whose 0x7F / 0xFF encodings are NaN and NaN != NaN --
  the same "failure" appeared baseline vs baseline; the gate now compares bytes.)
* `tests/test_frontend_fe78.py --mma cc --seeds 40 --rows 1 2` (cc/lean vs the legacy 96 x 4 WMMA path): PASS,
  240 row-evaluations, 0 top-8 index-set mismatches, 0 weight diffs > 1e-6, 0 x / x_sf mismatches.
* 8-rank gates (`tests/fe5_gates.sh`): all PASS (10.6.131.8, build fee8931, `/raid/kimi/results/fe5/gates.log`):
  * `tests/test_select_in_mega.py` (DG_FE_CC_LEAN default 1), 50 seeds, 8 ranks: mxfp4 M2 / M16, qoq M2 / M16: 0 topk_idx
    and 0 topk_weights mismatches (400 / 800 rows per cell), y max |dy| 0, cos_min 0.9999998.
  * `tests/test_four_api_correctness.py --frontend fe --router-ref torch`, fused mxfp4 + qoq, T = 1 / 2 / 8 / 16 / 32
    tokens per rank (M = 8 .. 256), DG_FE_CC_LEAN=1 and =0: the FE's top-8 index sets agree with the pure-torch router
    reference (bf16 GEMM -> bf16 logits -> top-8 -> fp32 softmax) on 8/8, 16/16, 64/64, 128/128, 256/256 tokens,
    weight max |diff| 6e-8 .. 1.2e-7, mxfp4 x bytes identical to the torch per-token cast (0 differing bytes), qoq x
    4 / 4 / 37 / 89 / 155 differing int8 bytes of 8T x 3072 (the FE's `__float2int_rn` vs the torch cast on exact
    .5 ties; identical for LEAN 0 and 1, pre-existing); y vs the torch-routed dequantised reference: mxfp4 cos_min
    0.99999 / 0.99999 / 0.99992 / 0.99992 / 0.99992, qoq 0.99993 / 0.99993 / 0.99993 / 0.99992 / 0.99992, norm ratio
    0.9998-1.00004; every metric identical to the last digit between LEAN=1 and LEAN=0. T=32 also with --slot-check
    (clean), and the balanced `--tokens 32 --hot-rows 12 --slot-check` (mxfp4 0.99996, qoq 0.99993). T = 1 and 2 are
    the cells that run the cc / lean router (rows <= 2 per rank); T >= 8 run the swapab router.

## 5. What remains

* The lean kernel is bounded by the weight-stream latency (long_scoreboard): the 2.36 MB router matrix is read from
  DRAM every launch because the Mega's weight stream evicts it (round 4: the persisting-L2 window does not hold it).
  Prefetching it into L2 from the tail of the preceding Mega kernel (idle epilogue warps, `cp.async.bulk.prefetch` /
  `prefetch.global.L2`) is the remaining FE lever that does not require fusing the FE into the Mega; not tried here.
* The nsys span carries ~0.7-0.9 us above the in-kernel chain (launch ramp / last-CTA drain); a smaller CTA count
  or PDL were negatives in earlier rounds.
* `select_pruned` could only pay off if the select were on a critical path, which in the pipeline it is not.
