# Fable cc router, round 5 (2026-09-12, H20-3e 10.6.131.7, branch perf/phase-stamps-probe)

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

RESULTS_PLACEHOLDER

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
* 8-rank gates (`tests/fe5_gates.sh`): GATES_PLACEHOLDER

## 5. What remains

* The lean kernel is bounded by the weight-stream latency (long_scoreboard): the 2.36 MB router matrix is read from
  DRAM every launch because the Mega's weight stream evicts it (round 4: the persisting-L2 window does not hold it).
  Prefetching it into L2 from the tail of the preceding Mega kernel (idle epilogue warps, `cp.async.bulk.prefetch` /
  `prefetch.global.L2`) is the remaining FE lever that does not require fusing the FE into the Mega; not tried here.
* The nsys span carries ~0.7-0.9 us above the in-kernel chain (launch ramp / last-CTA drain); a smaller CTA count
  or PDL were negatives in earlier rounds.
* `select_pruned` could only pay off if the select were on a critical path, which in the pipeline it is not.
