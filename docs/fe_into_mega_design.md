# Fusing the Fable frontend into the SM90 fused MegaMoE kernel (`DG_FP4_FUSE_FE`)

Tiny-M only (global tokens <= 16, i.e. 1-2 activation rows per rank). Default 0 until it wins.

## Today (knob 0)

E2E = `router_quant_topk_kernel` (csrc/fable_frontend.cu, `DG_FE_TINYM` path: 96 router CTAs =
24 groups of 16 experts x 4 K-parts of 768, plus `m` quant CTAs; 256 threads each, 3 CTAs/SM,
single wave) -> gap -> `sm90_*_mega_moe_h20_fused_impl` (78 persistent CTAs x 384 threads).
FE kernel 6.9 us: ~1.5 us launch ramp + first HBM round trip, ~0.5 us final writes of
`topk_idx / topk_weights / x / x_sf` that the Mega dispatch warps read back; gap 0.3-1 us.

## Fused (knob 1): who computes what

Kernel args added (all `nullptr` when the knob is off): `fe_hidden` (bf16 [m, H]),
`fe_router_weight` (bf16 [E, H]), `fe_workspace` (the Fable frontend workspace tensor, see below).
Template flag `kFuseFERequested` -> `kFuseFE`.

The **FE crew** of every CTA is its 8 math warps (`kNumEpilogueThreads` = 256 threads, epilogue
role), which are idle until the first math task is published. They run the frontend right after
`warpgroup_reg_alloc<208>` and before their first "sync with dispatch"
(`kDispatchWithEpilogueBarrierIdx`). Named barrier `kFEBarrierIdx` = 10 (256 threads) replaces
the FE kernel's `__syncthreads()`.

| step | who | what |
| --- | --- | --- |
| 1 | FE crew of CTA `c` | issue the loads of its router units: one 512 B `cp.async.bulk` per (unit, chunk, row) on one transaction mbarrier, all units at once (v1 used 32 x 16 B `cp.async` per row: LSU issue-bound, 2.3 us) |
| 2 | FE crew of CTA `t < m` | quantise token `t` (`quant_role`, MXFP4: fp8 e4m3 + per-K128 fp32 SF, QoQ: int8 + per-row scale) into `buffer.x / x_sf` while the router loads are in flight |
| 3 | FE crew of CTA `c` | mbarrier wait, barrier, WMMA of each unit (identical to the FE router CTA), store the fp32 partial logits `[k_part][m][E]` into the workspace; barrier, thread 0 `atom.release.gpu.add router_done += 1` (cumulative over the crew's stores, like the push DONE ticket) |
| 4 | FE crew of CTA `t < m` | thread 0 spins `ld.acquire.gpu router_done >= 78`, barrier, `topk_softmax_token_tiny` for token `t` (all 256 threads fetch the 4 partials per expert, warp 0 selects the top-8 + softmax), write `topk_idx / topk_weights`; barrier, thread 0 `atom.release.gpu.add topk_done += 1` |
| 5 | dispatch warps of every CTA | lane 0 spins `ld.acquire.gpu topk_done >= num_tokens`, `__syncwarp`, then today's routing (push dispatch) unchanged, except that `topk_idx / topk_weights / x / x_sf` are read with `ld.global.cg` (`__ldcg`) instead of `ld.global.nc` (`__ldg`): they are produced inside this kernel by other SMs |

Router unit `u` (0..95: expert group `u / 4`, K-part `u % 4`) runs on CTA `kNumSMs - 1 - (u % kNumSMs)`:
CTAs 77..60 carry two units (u, u + 78), CTAs 59..0 one; the top-k CTAs 0..m-1 (m <= 16) therefore
carry one unit. Inside a unit the mapping is the FE's: warp `w` (0..7) owns the K-slice
`[32w, 32w + 32)` of every 256-wide chunk, `m_tile` = 0 (m <= 16), chunks 0..2 of the K-part in
order, `wmma m16n16k16 bf16 -> fp32` with the same fragments in the same order, per-warp partials
reduced in warp order 0..7, K-part partials summed in order 0..3 by the top-k, then bf16-rounded.
Everything is the same device code (extracted from csrc/fable_frontend.cu into
deep_gemm/include/deep_gemm/impls/fable_frontend_device.cuh, which the .cu now includes), so the
outputs are bit-identical to the standalone FE.

## Hand-off buffers

The Fable frontend workspace tensor (`fable_frontend_workspace_bytes(E)` bytes, zero-initialised
once, cached on the symmetric buffer object): bytes `[0, 256)` counters (the standalone FE uses
words 0/1; the fused path uses words 8 = `router_done`, 9 = `topk_done`), then the fp32 partial
logits `[4][64][E]` (the FE layout `logits + (k_part * m + t) * E + e` with the launch's `m`).
Per-launch reset: SM0's dispatch thread 0 zeroes words 8/9 in `cleanup_workspace` (after the
dispatch grid sync of the launch, i.e. after every reader of both counters; the next launch on
this stream starts after the kernel completes, so no epoch scheme is needed).

## Shared memory budget

The BM8 fused kernel is launched with the full 232448 B (`SM90ArchSpec::smem_capacity`); its
static layout is ~188 KB: expert counts 2 KB, send buffer 6 KB, LUT 1 KB, CD 5 KB, 4 stages x
(2 KB A + 40 KB packed B) = 172 KB, SFA 1 KB, barriers / scheduler mailbox < 1 KB. The 44 KB of
slack is smaller than the FE router's 59 KB, so the FE crew borrows the **pipeline stage region**
(`smem_a[0]` .. `sf_start_ptr`, 172032 B contiguous, 1024 B aligned): it is untouched until the
first task is published, which is causally after every rank's routing, hence after all FE work
(the loaders only fill a stage after `claim_next_task`, which needs the DONE flags of all ranks,
which need every CTA's pushes, which wait for `topk_done`, which waits for `router_done == 78`).

FE crew usage: 2 units x 58880 B (3 stages x 32 rows x 264 bf16 = 50688 B + 8 warps x 256 fp32
partials = 8192 B) = 117760 B, + 128 B quant warp maxima + 2048 B top-k keys + 16 B transaction
mbarrier = 119952 B (static-asserted <= the stage region). No change to the kernel's smem layout
or launch size. The crew itself lives in `fused_fe_crew` (`__noinline__`, sm90_fp4_mega_moe_h20_fused.cuh).

## Expected gain

Removes the FE launch ramp, the FE -> Mega gap and the write-out/read-back of the frontend outputs
(-2..3 us E2E expected); the router's HBM round trip and the top-k now overlap the Mega prologue
(TMA descriptor prefetch, barrier init, the dispatch warps' DONE-count reads).

Costs added to the Mega critical path: one HBM round trip for the router weights + 3 chunks of
WMMA per unit (2 units on 18 CTAs), the grid-wide `router_done` counter (78 release atomics on one
word), the top-k partial fetch (one L2 round trip), the `topk_done` release/acquire. Dispatch warps
that used to start routing at kernel entry now wait ~3-4 us for the frontend.

## Gating (host, csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp)

`fuse_fe` requires the frontend tensors, `DG_FP4_FUSE_FE=1`, the interleaved scheduler,
`num_tokens <= 16` and `<= kNumSMs`, `topk == 8`, `E % 16 == 0`, `E <= 512`, `H % 1024 == 0`
(H / 256 / 4 chunks per K-part, 3 for H = 3072 == the FE's 3 stages), a bf16 contiguous hidden
`[num_tokens, H]` and router weight `[E, H]`, and a workspace of >= 256 + 4 * 64 * E * 4 bytes.
The kernel name gets the `_fefuse` suffix. `kFuseFE` static-asserts `!kTinyMGemv` (that path's
math warps start streaming right away).

## Python

`deep_gemm.mxfp4_mega_moe_fused / qoq_mega_moe_fused(..., frontend=(hidden, router_weight))`
runs the fused path (the workspace comes from `deep_gemm.fable_frontend_workspace(buffer, E)`);
without `frontend` the kernel is today's. `tests/profile_four_api_h20.py` / `bench_frontend_tinym.py`
skip the FE launch and pass `frontend=` when `DG_FP4_FUSE_FE=1` (the E2E graph span is then just
the fused kernel). `tests/test_four_api_correctness.py --frontend fused` runs the standalone FE +
Mega first (reference routing / quant / y), then the fused kernel, and asserts bit-equality of
`x / x_sf / topk_idx / topk_weights / y` plus the usual exact-quantised reference check.

## Result (H20 .7, 2026-09-11, 8 ranks, tip perf/fe-into-mega)

**Correctness.** `tests/test_four_api_correctness.py --frontend fused` (standalone FE + Mega first,
then the fused kernel into cleared views): `x / x_sf / topk_idx / topk_weights / y` bit-identical on
all 8 ranks for T=1 (M=8) and T=2 (M=16), both quants; exact-reference cos_min MXFP4 0.999996 /
0.99999, QoQ 0.99993 with the FE's real (unbalanced) routing. The knob is inert without frontend
tensors (`--tokens 2 8 8 16` matrix unchanged: MXFP4 0.99997-0.99999, QoQ 0.99993). 200-iteration
graph-replay stress (FE / Mega / FE+Mega / FEinMega graphs) clean at M=2/8/16, both quants.

**Side finding (fixed, commit "split-K tail tasks of a wave-scheduled launch ...").** With the FE's
real routing the *existing* kernel was wrong on ranks with >= 8 active local experts at <= 8 global
tokens (cos_min 0.67; per-slot cos 0.15-0.3 on the highest-index experts of those ranks): stream-K is
compiled in for M <= 8 but inactive in-kernel once the rank has >= 78 L1 tasks, and the wave
scheduler's 2-way split-K tail tasks then took the stream-K reduction branch (worker slots 0/1 shared
by every concurrent tail). The balanced correctness test never produced >= 78 L1 tasks with stream-K
compiled in. Non-stream-K tasks now carry `first_worker_idx == kNotStreamKWorker` and use the split-K
publisher/finisher protocol; `--frontend fe` (real routing, knob off) is the regression test.

**Performance: a loss at every M (default stays 0).** CUDA-event E2E (`tests/bench_frontend_tinym.py`,
n=100, host barrier, GPU0 median us; FE+Mega graph = knob 0, FEinMega graph = knob 1, same process):

| variant | MXFP4 M2 | QoQ M2 | MXFP4 M8 | QoQ M8 | MXFP4 M16 | QoQ M16 |
|---|---|---|---|---|---|---|
| v1 cp.async loads, inlined crew: FE+Mega -> FEinMega | 80.2 -> 86.5 (+6.3) | 77.5 -> 85.2 (+7.6) | 80.2 -> 86.0 (+5.8) | 78.2 -> 85.6 (+7.4) | 116.8 -> 124.1 (+7.3) | (n/a) |
| v2 bulk copies, prefetching top-k, inlined | 83.7 -> 91.3 (+7.5) | 79.2 -> 86.7 (+7.5) | 80.8 -> 86.9 (+6.1) | 79.7 -> 86.1 (+6.4) | 101.3 -> 107.9 (+6.7) | 97.7 -> 102.9 (+5.3) |
| v4 = v2 + crew in a `__noinline__` function (final) | 95.2 -> 100.0 (+4.8) | 94.8 -> 99.7 (+5.0) | 79.9 -> 84.7 (+4.8) | 78.3 -> 84.7 (+6.4) | 115.4 -> 120.2 (+4.8) | 99.0 -> 103.9 (+4.9) |

(Absolute numbers drift between sessions -- clocks not locked in the bench, another agent's
single-GPU jobs on the node; the knob-0/knob-1 pair of each row is from the same process.)

Where the time goes (rank-0 phase stamps, MXFP4 M=8, same FE routing for both, us from kernel
entry, `tests/profile_fused_phase_stamps.py --fe-routing` vs `--fuse-fe`):

| | knob 0 (FE launched before, routing from FE) | knob 1 v2 (inlined) | knob 1 v4 (noinline) |
|---|---|---|---|
| init done | 1.47 | 1.47 | 1.50 |
| FE loads issued (max) | - | 5.06 | 7.01 |
| FE loads landed (max) | - | 5.58 | 7.52 |
| FE WMMA + partial stores (max) | - | 7.23 | 9.17 |
| FE top-k written (max) | - | 10.48 | 15.63 |
| routing done (DONE flags) | 10.69 | 19.68 | (skewed session) |
| first math task | 14.00 | 22.98 | |
| last L1 task end | 25.36 | 37.97 | |
| kernel end | 34.61 | 48.43 | |
| L1 stage head-to-head (SM0, ns) | 983 | 1747 | 1074 |
| L1 stage RF decode + LUT (ns) | 574 | 1213 | 831 |
| SM0 L1 task (us) | 10.61 | 14.21 | 10.61 |

Per-CTA durations (max over CTAs, v2/v4): issue 3.2/4.8 us, land 1.3/1.1, WMMA+store 1.7/1.9,
release 0.45/0.5, top-k wait-for-units 2.0/2.7, top-k compute+write 2.5/5.5.

Three reasons the fusion loses:
1. The crew is slower than the standalone FE at every step. Issuing the router loads costs 2.3 us
   (cp.async, LSU issue-bound) or 3.2-4.8 us (bulk copies -- not better), the data lands within
   ~1 us of the last issue, i.e. the standalone FE's 96 CTAs x 3 CTAs/SM issue far more in parallel
   than 78 x 8 warps (18 CTAs carry 2 units). WMMA + partial stores 1.7 us, top-k 2.5-5.5 us (the FE
   kernel: ~1.5 us). The dispatch sees the top-k at 11 us (v2) after kernel entry; the standalone FE
   + gap costs the graph ~9-10 us (FE+Mega - Mega = 8.9-10.2 us) and overlaps the Mega launch ramp.
2. Inlined into the math role the crew's code (WMMA fragments, bulk-copy / mbarrier asm, top-k)
   degrades the RF K-loop's code generation: the per-stage decode+LUT doubles (574 -> 1213 ns),
   the L1 phase +3.6 us and the L2 tail +1.5 us at identical routing (0 spills reported). A
   `__noinline__` crew restores the K-loop (SM0 L1 task 10.61 us both ways) but makes the crew
   itself slower (ABI call, 15.6 us to the top-k).
3. The dispatch warps idle for the whole FE phase, whereas with a separate FE kernel the Mega
   prologue (TMA descriptor prefetch, barrier init, DONE-count reads) already overlaps the FE tail.

What would be needed to win (not done): a crew that finishes the top-k <= ~5 us after entry --
e.g. all 4 K-parts of an expert group on one CTA with a per-group candidate top-8 (24 CTAs x 96 KB,
bandwidth-bound ~1 us, hand-off of 192 keys instead of 1536 partials), the quant on the loader
warps, and the crew code kept out of the math role's register allocation without an ABI call
(separate warps: not available -- the 8 math warps are the only 256-thread group with registers).
Customer-method nsys captures (knob 0 vs 1, e2e fused, M=2/8/16, 3 each) were NOT obtained: on
2026-09-11 the node was shared with another agent's jobs (single-GPU FE experiments, later an 8-GPU
job) and `scripts/capture_four_api_h20_timelines.sh` correctly refuses / drops reports when another
GPU process appears. When the node is exclusive:
`for p in 1 2 3; do bash scripts/run_fe_fuse.sh capture /raid/kimi/results/fefuse/cap_k0/p$p "2 8 16" -- DG_FP4_FUSE_FE=0; bash scripts/run_fe_fuse.sh capture /raid/kimi/results/fefuse/cap_k1/p$p "2 8 16" -- DG_FP4_FUSE_FE=1; done; python3 scripts/summarize_knob_captures.py --knob 0 .../cap_k0/p* --knob 1 .../cap_k1/p* --fused-only`.
The CUDA-event A/B above (two independent passes, +4.8..+7.6 us at every point) already decides the
default; the customer method measures the same graph span (frontend + MegaMoE).
