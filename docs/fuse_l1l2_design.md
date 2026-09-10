# DG_FP4_FUSE_L1L2: two-layer fusion for tiny M (H20 fused MegaMoE, BM8 RF swapAB)

Knob `DG_FP4_FUSE_L1L2` (host, default 0; gate `DG_FP4_FUSE_L1L2_MAX_M`, default 16 global
tokens). Kernel flag `kFuseL1L2` (body): MXFP4/QoQ, BM8, BN256, dense tiles, interleaved
scheduler, 2 K128 blocks per stage, exclusive with half-tile / L2 half-row / split-K L2 /
stream-K / tiny-M GEMV (the host forces `split_k_l2_ways = 0`, `stream_k = 0` under the knob).

## Unit
An L1 task (expert e, L1 N-block n of 10, full K = 24 K128 blocks = 12 stages) keeps its
SwiGLU output (128 intermediate columns == L2 K-block n, since `kNumL1BlocksPerL2KBlock == 1`)
in SMEM and immediately runs the **W2 K-slice**: 12 output N-blocks j (256 W2 rows each) x
K128 block n = 12 dense 20 KB tiles = 6 more pipeline stages (2 tiles per stage). Per tile:
RS decode (existing MXFP4 LUT / QoQ zsub decoders, 2 halves x 64 rows per WG), 4 K32 RS
WGMMAs against the SMEM intermediate (B operand), promote with the intermediate SF exactly
as today's L2 `promote_stage_rf` for one K128 block (QoQ: x s2[row] from packed byte 64),
giving an fp32 partial out[8 tokens x 256 rows]. Partials are reduced across the 10 L1 tasks
of the same pool block with `red.global.add.f32` into a per-(pool block, j) fp32 scratch slot
([element][256 threads] layout, 8 KB; same as the split-K partial layout), then one
`atom.acq_rel.gpu.add` ticket per (pool block, j): the 10th arriver (acquire) loads the slot
(`ld.global.cg`), zeroes it, resets the ticket, and runs today's L2 swapAB epilogue for N-block
j (bf16 x l2_row_scale -> smem -> NVLink scatter -> fine-combine mailbox entry).
No L2 tasks exist: the scheduler's L2 task count is 0 (`kNoL2Tasks` template flag: all L1
waves are "warm-up" waves), the intermediate global buffer, `notify_l1_ready` /
`l2_arrival_mask` and the L2 SF buffer are unused (the L1 epilogue skips the TMA store, the
global SF store and the notify under the knob; the cleanup still zeroes the mask, harmless).
Tile order per task: j = (u + n) mod 12 for u = 0..11 (rotation by the L1 N-block), so the
10 contributors of a pool block arrive at different j last and the 12 finisher epilogues
spread over up to 10 CTAs instead of all landing on the last one.

Split-K L1 tail (last partial L1 wave, 2 K halves): the finisher half (k_split 1) owns SwiGLU
today and therefore also owns the W2 slice; the publisher half does nothing extra. All three
roles derive `has_w2 = L1 && (num_k_splits == 1 || k_split_idx == num_k_splits - 1)` from the
same task payload, so the per-task stage count (12 or 6, + 6) agrees.

## SMEM plan
* Intermediate B tile: new dedicated 2 KB region `SMEM_FUSE_SIZE` after the CD region
  (1 KB, 1024 B aligned: the [8 tokens x 128 K] int8/fp8 tile in the TMA SW128 layout the
  L2 A-loader produces today: byte (t, k) at t*128 + (((k>>4) ^ (t&7))<<4) + (k&15); then
  8 floats of per-token SF (`sf_pair.x`, the value written to the L2 SF buffer today);
  rest padding). Written by the 256 epilogue threads from `smem_cd_l1` (row-major [8][128])
  after the FP8/INT8 quantisation, then `fence.proxy.async` + CTA barrier before the first
  W2 WGMMA. Lives for the whole W2 slice; the CD region is free for the finisher epilogues
  (`smem_cd_l2` bf16 [8][256]).
* Stage ring: unchanged (4 stages x (2 KB A + 40 KB packed B + 256 B SF)). W2 stages use
  only the packed-B slot: the B loader issues two 20 KB bulk copies (tiles j0, j1 at +0 /
  +20 KB), the A loader only `arrive()`s (no A/SF bytes). The loaders run kNumStages ahead,
  so W2 stages 0..3 are resident while the L1 K loop / SwiGLU finish.
* Tickets: the 16 B stream-K ticket word doubles as a [2 parity][2 blocks] ticket broadcast.
* Budget (BM8, 4 stages): 2 + 7 + 1 + 5 (CD) + 2 (new) + 173 (stages) + 1 (SF) + ~0.4
  (barriers) ~= 191.4 KB < 227 KB.

## Scratch / ticket (workspace)
Reuse the L2 split-K slots (unused under the knob): scratch slot
`get_splitk_l2_scratch_ptr(pool_block, j, 0)` (96 pool blocks x 12 x 8 KB = 9.4 MB of the
existing 18.4 MB L2 region), ticket `get_splitk_l2_flag_ptr(pool_block, j)`. The workspace is
`torch.zeros` at allocation; every used slot is zeroed by its finisher after the read and the
ticket reset to 0 there, so the next launch starts from zero (launch parity: launch N+1's
first red.add happens after every rank passed NVLink barrier #3 of launch N, i.e. after all
math tasks, hence after the finisher's reset; the kernel boundary orders the stores). The
SM0 cleanup also zeroes all flags (unchanged).
Protocol per stage (2 tiles): red.add both partials -> `bar.sync` (256) -> warp 0 lanes 0/1
issue the two `atom.acq_rel.gpu.add` tickets (results consumed one stage later, so the L2
round trip is hidden behind the next stage's math; after the last stage a final barrier
drains them) -> tickets broadcast through smem -> every thread runs the finisher epilogue
for each j whose ticket == 9 (bar.sync + acquire by thread 0 is cumulative over the other
threads' red.adds, as in the split-K / stream-K finishers; the slot is read with ld.cg).

## Epilogue reuse
The L2 swapAB epilogue (bf16 x per-row scale into `smem_cd_l2`, WG barrier, 16-lane row
scatter over NVLink, CTA barrier, `signal_combine_arrivals`) is factored into
`l2_epilogue_swap(l2_n_block_idx)` reading `final_accum`; the existing L2 task path calls it
with its own n_block, the finisher fills `final_accum[h*32 + i]` from the slot and calls it
with j. The mailbox entry / combine target are unchanged (12 entries per pool block).

## Numerics
Same requant bytes and SF as today (the tile is the same bytes the L2 A-loader would have
fetched), same per-K128 promote (int32/fp32 x SF x s2), same bf16 cast x l2 row scale. Only
the association of the 10 fp32 partials changes (red.add order) -> run-to-run ulp-level
differences; cos_min / norm_ratio targets unchanged (>= 0.99999 MXFP4, >= 0.99993 QoQ).

## Expected timeline (M8, 78 SMs) and the tail caveat
Task = 12 L1 stages (~18 us) + 6 W2 stages (~7.5 us) ~= 26 us. The 96 x 7.1 us L2 tasks
(682 SM-us, 22 CTAs ran a second one) become 80 x 7.5 us slices inside the L1 tasks, so the
L2 quantisation loss (~6 us) and the L1->L2 dependency waits disappear. Caveat: the 2 tail
tasks (80 on 78 SMs) now carry their W2 slice too: the split-K finisher half runs 6 L1
stages (waiting on its publisher) + SwiGLU + 6 W2 stages after wave 1 ends, i.e. the tail is
~12 stages (~17 us) instead of ~6 + the L2 fill. Whether the wave-1 gain outweighs the
longer tail is exactly what the A/B measures; M=2 (one wave, no tail) is the cleanest case.

## Probe
Slot 4 = last L1 task end (now includes the W2 slice); slot 5 = last finisher epilogue end;
slot 41 = finisher epilogue count, 42 = finisher epilogue SM cycles (SM0), 43 = W2 slice
count, 44 = W2 slice SM cycles (SM0, first W2 full-barrier wait to last ticket).

## Result (H20, 8 ranks, tip b323af2, 2026-09-10)
Correctness (DG_FP4_FUSE_L1L2=1, DG_FP4_SPIN_TIMEOUT=1): MXFP4 T=2 cos_min 0.99999, T=8 x3
0.99999 (bit-identical runs), T=16 x2 0.99998, norm_ratio 0.99994-0.99997; T=128/512 (knob
inactive, untouched path) 0.99959/0.99989 as before; QoQ T=8 0.99993, T=16 0.99993, norm_ratio
1.00003-1.00004. 200-iteration graph-replay stress at M=8 (MXFP4 + QoQ): no trap / hang.
Mechanism (probe task log, M=8 MXFP4): 78 CTAs, 74 x 1 task + 4 x 2 (78 full + 4 split
halves), zero L2 tasks, 96 finisher epilogues = 8 pool blocks x 12.

Timing: it LOSES at every M, by 2x at M >= 8. Skew-free min-over-devices (host barrier, nsys,
2 passes, knob 0 -> 1, us): MXFP4 M2 41.1/41.4 -> 51.6/52.3, M8 55.2/55.4 -> 105.0/106.9,
M16 77.5/78.7 -> 146.8/139.6; QoQ M8 54.1/54.7 -> 101.6/103.8, M16 75.5/75.7 -> 151.1/144.8.
Why (per-CTA task log + slots 41-44, M=8 MXFP4): a fused L1 task takes 36 us p50 / 44.6 p100
instead of 19.4 (L1) + 7.7 (a whole L2 task): SM0's W2 slice costs 21.8 us, of which 8.8 us
are its finisher epilogues (slot read + bf16 + scatter + mailbox, ~1.5-2 us each) and ~13 us
the 12 tiles (1.08 us per tile vs 0.77 per L2 K-block: red.add of 8 KB per tile, the
per-stage CTA barrier + tickets, no cross-stage decode overlap). The split-K tail finisher
half then runs 45 us (70.5 -> 115.3): 6 L1 stages + SwiGLU + the W2 slice + ALL 12
finisher epilogues, because it is by construction the last arriver of its pool block (the
j-rotation only spreads finishers when the 10 contributors finish within ~12 tile times).
At M=2 every task is a split half, so the tail is the whole phase (43.7 vs 25.5 us last L1).
The structural point: the unfused scheduler overlaps the L2 tasks of finished pool blocks with
other CTAs' L1 tasks and leaves only ~5-7 us of L2 after the last L1, while fusion puts the
whole L2-equivalent work (plus the reduction protocol) on every L1 task's critical path and
concentrates the epilogues on the stragglers. Default stays 0; the knob is kept as a
documented negative result (see the host comment for the customer-method numbers).
