# Tiny-M CUDA-core GEMV math path for the SM90 fused MegaMoE (`kTinyMGemv`)

Host-selected when `num_tokens_per_rank * num_ranks <= DG_FP4_TINYM_MAX_M` (default 16) on the
BM8 MXFP4 / QoQ RF-swapAB tier (`DG_FP4_TINYM=1/0`, default 1). Everything outside the L1/L2
math is unchanged: routing, expert-count bcast, dispatch pull into the recv pool, the
`l1_arrival_count` / `l2_arrival_mask` dependency words, the L1 output pool + per-token L2
SF, the L2 NVLink scatter + fine-combine mailbox, combine, workspace cleanup. The TMA A/B
loader warps and the interleaved task scheduler are idle (`if constexpr (!kTinyMGemv)`);
the 8 math warps (256 threads, 208 regs) run `sm90_fp4_mega_moe_h20_tinym_math.inl`.

## Why
At M = 2/8/16 global tokens a rank holds 1-2 rows per active expert; the tensor-core BM8
path costs ~650 ns per K128 block of fixed TMA/mbarrier/wgmma skeleton, so L1 takes 16/30/48 us
against a weight-streaming floor of ~4/14/27 us. The GEMV path is bandwidth-shaped: every
lane keeps >= 8 x 16 B weight loads in flight and does ~11 (MXFP4) / ~3 (QoQ) integer ops
per 8 nibbles plus HFMA2 / DP4A.

## Work unit and partition (stream-K)
* Unit = (pool_block p, weight tile n, K128 block k) = one dense 20480 B packed tile
  (256 rows x 80 B). L1: 10 tiles x 24 K-blocks per pool block; L2: 12 x 10.
* Linear order u = (p * NB + n) * NKB + k. CTA i owns the contiguous range
  [floor(i*U/78), floor((i+1)*U/78)) per phase (L1 phase, then L2 phase). Within the CTA all 8
  warps walk the same units; warp w owns rows [32w, 32w+32) of every unit, lane l owns rows
  32w + 8r + (l>>2), r = 0..3, and the 16 B chunk c = l & 3 (4 lanes per 80 B row: chunk c =
  RF word c of K32 groups 0..3 after the host word transpose; +1 x 4 B meta at byte 64).
* Tile (p, n) accumulates over its K-blocks in fp32 registers (acc[4 rows][8 tokens]).
  When the CTA leaves a tile: 4-lane shuffle reduce (2 xor steps). If the CTA covered
  k = 0..NKB-1 it runs the epilogue directly. Otherwise it `red.global.add.f32`s its partial
  into the tile's 8 KB scratch slot ([256 rows][8 tokens] fp32, the existing split-K slots
  `get_splitk_l{1,2}_scratch_ptr(p, n)`), `bar.sync`, and thread 0 takes a ticket
  (`atom.acq_rel.gpu.add` on `get_splitk_l{1,2}_flag_ptr(p, n)`). The last arriver
  (ticket == num_splits - 1, num_splits derived from the partition arithmetic, identical on
  every CTA) adds the slot to its registers (ld.cg), re-zeroes slot + ticket (self-cleaning
  for the next launch), and runs the epilogue. Non-last arrivers never wait. Splits per tile
  <= ceil(NKB / units_per_cta) + 1 (<= 7 at 1 pool block, 2 at M = 16). Requires
  pool blocks <= 64 (kSM90SplitKL1MaxPoolBlocks); the host gate guarantees it.
* Deadlock freedom: L1 units only wait on token arrival; L2 units wait on the L1 bits of
  their pool block; no unit waits on a later unit.

## Activation staging (smem, per (phase, pool block), CTA-wide)
Stage region = the idle pipeline stages (`smem_a[0]`, >= 168 KB). MXFP4: fp8 -> fp16 via
`cvt.rn.f16x2.e4m3x2` scaled by 2^-6 (exact), [8 tokens][K] halves (48 KB). QoQ: int8 copy
(24 KB) + per-(token, K-block) int32 sum of the 128 activations. Per-(token, K-block)
fp32 SF (x 4096 for MXFP4, see below). MXFP4 also holds a 16-entry x 8 B LUT: hi byte of
fp16(E2M1 magnitude m x 2^(s-15)) for LUT index s (host relative index, 0 = flush).
L1 waits `l1_arrival_count[p] == valid_m`; L2 waits all NKB bits of `l2_arrival_mask[p]`.

## Inner loop (per lane, per unit: 4 rows, 1 K128 block)
Loads: 4 x `ld.global.nc.L1::no_allocate.v4` + 4 x u32 meta per unit, issued
`kTinyMPrefetch` (default 2, `DG_FP4_TINYM_PREFETCH`) units ahead -> >= 8 x 16 B in flight
per lane, ~40 KB per SM.
* MXFP4 word (group g, chunk c; braided RF order): `sel = w & 0x77777777`,
  `hb_hi = prmt(lut, sel)`, `hb_lo = prmt(lut, sel >> 16)` (fp16 hi bytes of K g*32+4c+b and
  g*32+16+4c+b), signs `|= w & 0x80808080` / `|= (w << 4) & 0x80808080` (same as the fp8
  decoders), then `prmt(hbs, 0, 0x1404 / 0x3424)` expands to fp16x2 pairs. 4 HFMA2 per word
  per token against the staged fp16 activations (2 x 8 B LDS per group per token), two
  fp16x2 chains per row; per unit the half2 sum is promoted once per (row, token):
  `acc += (sf * 4096) * (float(h.x) + float(h.y))`. Range: |w| <= 6, |a| <= 448 * 2^-6 = 7,
  16 products per half -> |sum| <= 672, no fp16 overflow; 6-bit products are exact, the
  fp16 accumulation error over 16 terms is ~2^-12 relative (cos >= 0.9999 target).
  Scale bookkeeping: kernel fp8 LUT = m * 2^(s-9) = 2^6 * (m * 2^(s-15)); activation 2^-6
  -> total 2^12 = 4096 folded into the SF.
* QoQ word: `hi = (w >> 4) & 0x0f0f0f0f`, `lo = w & 0x0f0f0f0f` (unsigned codes), 2 x
  `dp4a.u32.s32` per word per token against staged int8 activations; per unit:
  `v = acc - z[row] * sumA[token][k]` (exact int32, == sum (code - z) * a),
  `f = magic_i2f(v)`, `acc += (sf[token][k] * s2[row]) * f` (same association as the RF path).

## Epilogues (last arriver, CTA-wide)
* L1 tile (p, n): rows 16q..16q+7 = gate, +8 = up for output columns n*128 + 8q + j; lane
  (w, l) holds gate/up pairs for columns 16w + 8(r/2) + (l>>2), tokens t = c, c+4.
  v = silu(clamp(g*s1_g)) * clamp(u*s1_u) * topk_w[t]; per-token amax over the 128 columns
  (shuffle over the 8 lanes sharing c, then smem over the 8 warps) -> one SF per token
  (e4m3 `get_e4m3_sf_and_sf_inv`, QoQ amax/127) -> byte tile [8][128] in `smem_cd_l1` ->
  the existing `tensor_map_l1_output` TMA store, SF to `l2_sf_buffer[n * padded + m_idx + t]`,
  `fence.proxy.async`, `notify_l1_ready(p, n)`.
* L2 tile (p, n): bf16(acc * l2_row_scale[n*256 + row]) -> `smem_cd_l2[t][256]` -> warp t
  scatters 512 B to `combine_token_buffer(topk)[token] + n*512` on the destination rank
  (16 B per lane), bar.sync, fine-combine mailbox post (or nothing on the barrier path).

## Scratch / flags reset
Slots and tickets are zero at buffer creation; the last arriver re-zeroes what it consumed;
the existing SM0 cleanup also zeroes the ticket words in TinyM mode. Rank-local only.

## Duplication note
The stream-K scheduler another agent is adding (`DG_FP4_STREAMK`, scheduler/mega_moe_fused.cuh)
had not landed when this was written; the partition here is ~20 lines of arithmetic inside
the new header (`tm_unit_range`, `tm_num_splits`) and can be swapped for the shared one.
