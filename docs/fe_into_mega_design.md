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
| 1 | FE crew of CTA `c` | issue the `cp.async` loads of its router units (all chunks of all units at once, one commit group) |
| 2 | FE crew of CTA `t < m` | quantise token `t` (`quant_role`, MXFP4: fp8 e4m3 + per-K128 fp32 SF, QoQ: int8 + per-row scale) into `buffer.x / x_sf` while the router loads are in flight |
| 3 | FE crew of CTA `c` | `cp.async.wait_group 0`, barrier, WMMA of each unit (identical to the FE router CTA), store the fp32 partial logits `[k_part][m][E]` into the workspace; `membar.gl`, barrier, thread 0 `atom.release.gpu.add` `router_done += 1` |
| 4 | FE crew of CTA `t < m` | thread 0 spins `ld.acquire.gpu router_done >= 78`, barrier, `topk_softmax_token_tiny` for token `t` (all 256 threads fetch the 4 partials per expert, warp 0 selects the top-8 + softmax), write `topk_idx / topk_weights`; `membar.gl`, barrier, thread 0 `atom.release.gpu.add topk_done += 1` |
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
partials = 8192 B) = 117760 B, + 128 B quant warp maxima + 2048 B top-k keys = 119936 B
(static-asserted <= the stage region). No change to the kernel's smem layout or launch size.

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
