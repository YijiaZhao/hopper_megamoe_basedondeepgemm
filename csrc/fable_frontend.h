#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Fused MegaMoE frontend for SM90: router logits (bf16 in, fp32 accumulate,
// bf16-rounded) + deterministic top-k + softmax over the selected logits +
// online activation quantization, written straight into the MegaMoE symmetric
// input views. `mode`: 0 = FP8 E4M3 per-token/K128 (NVFP4/MXFP4 weights),
// 1 = INT8 per-token whole-row scale repeated over the K128 slots (QoQ).
// `workspace` layout: [0,256) tickets, [256, 256 + 4*64*e*4) K-split partial
// logits, then kFrontendStampsBytes of optional per-CTA phase stamps
// (DG_FE_STAMPS=1). Zero-initialised once.
// `tiny` (DG_FE_TINYM): m <= 16 single-wave config (3 smem stages) -- same
// math, bit-identical outputs. `stamps_on`: record %globaltimer phase stamps.
// `l2_persist` (DG_FE_ROUTER_L2_PERSIST): 0 off; 1 = persisting-L2 set-aside +
// access-policy-window launch attribute over the router weights; 2 = PTX
// L2::evict_last cache hint on the router weight cp.async (no host set-aside).
// `pdl_mode` (DG_FE_PDL): 0 none; 1 = griddepcontrol.launch_dependents at CTA
// start; 2 = after each CTA's last store (the fused Mega is launched with
// programmatic stream serialization by the host when DG_FE_PDL != 0).
constexpr size_t kFrontendStampsOffsetBase = 256;
constexpr size_t kFrontendMaxCTAs = 256;
constexpr size_t kFrontendStampsBytes = kFrontendMaxCTAs * 8 * sizeof(unsigned long long);
// `grid` (DG_FE_TINYM_GRID, default 96): 96 = legacy tiny-M split (24 expert groups x 4
// K-parts + m quant/top-k CTAs); 0 = auto: full-K scheme sized to the SM count
// (H20: 77 router CTAs x 5 experts + 1 merger CTA = 78); N > 0 = full-K scheme
// with N CTAs in total. Full-K outputs are deterministic but not bit-identical to
// the legacy split (different fp32 accumulation order before the bf16 rounding).
size_t router_quant_topk_frontend_workspace_bytes(int e);
// `mma` (DG_FE_TINYM_MMA): 0 = WMMA bf16 m16n16k16 (default; legacy: cp.async smem ring,
// full-K: TMA row pieces); 1 = CUDA-core fp32 FMA straight from global (full-K only);
// 2 = swapab (legacy 96 x 4 grid and full-K): experts on the MMA M dimension, tokens on N,
// mma.sync m16n8k16 bf16 -> fp32, weight A fragments ld.global.nc straight into registers,
// activations staged once in smem; 3 = swapab with the router weights in A-fragment order
// (DG_FE_ROUTER_WLAYOUT=fragment, host permutes once: one warp load = one contiguous 512 B).
// H20-3e (.7) standalone, rows 1|2 x mxfp4|qoq, kernel-end stamp median (us, 5 stamped launches,
// 200-iter rerun; identical across the 4 cells unless a range is given):
//   96 x 4 wmma 6.91 | 96 x 4 swapab row 6.66 (-0.25) | 96 x 4 swapab fragment 6.14 (-0.77)
//   | 78 full-K wmma 7.94-8.45 | 78 full-K swapab 6.40-6.91.
//   NCU (96 grid, rows 1, mxfp4, --clock-control none): L2 read requests 49.1k (wmma) -> 37.7k
//   (swapab row, -23%) -> 19.0k (fragment, -61%); L2 read sectors 113.5k -> 77.0k -> 75.9k (-33%);
//   LSU instructions 89.0k -> 29.1k; kernel 9.41 -> 8.90 -> 8.42 us under NCU.
//   top-8 sets identical to the legacy WMMA path on 54k rows (1000 seeds x rows {1,2,8,16} x 2
//   quants) for swapab row AND fragment, 0 weight flips; 8-rank four-API cos_min unchanged.
// Default (python wrapper): mma = swapab, wlayout = fragment (the wrapper permutes a row-major
// router weight once per tensor and caches it; pass wlayout = 'pre' with a pre-permuted weight
// to make it a weight-transform-time cost). DG_FE_TINYM_MMA=wmma DG_FE_ROUTER_WLAYOUT=row = legacy.
// 4 = cc | 5 = cc6 (full-K grid, m <= 2, h = 3072, k_parts 1): CUDA-core K-split router, CTA =
// 5 experts x 4 (cc, 640 threads) | 6 (cc6, 960) warps, one warp = one expert x one K-part, every
// weight chunk ld.global.nc straight into registers as the kernel's first instructions (the m
// activation chunks issued back-to-back just before them). Hand-off (DG_FE_CC_MERGE, default
// ticket): keys stored -> bar.sync -> thread 0 atom.acq_rel.gpu ticket; the LAST router CTA reads
// the 616 key slots (5 x 16 B per lane), per-lane sorted top-8 + one 8-round redux merge
// (DG_FE_CC_SELECT=insert; =redux, 8 rounds of per-lane max + redux, measured 0.5 us slower),
// softmax, writes top-k. The (now merger-less) extra CTA only quantises the m rows. =poll keeps
// the streaming merger (16 B ld.cv polls; relaxed.gpu / cg polls measured 0.5-0.8 us slower).
// H20-3e (.7) standalone kernel-end stamp median (us), rows 1|2 x mxfp4|qoq, L2 flushed:
//   96 x 4 wmma 6.91 | 7.17 | 6.66 | 6.91  ->  cc ticket 4.61 | 4.61 | 4.86 | 4.86
//   (cc poll 4.86-5.38, cc6 poll +0.25-1.0). Chain (rows 1): chunk0 landed 1.54, all logits 2.05-2.30,
//   last CTA's keys + ticket 3.3-3.6, keys read +0.25, select +0.5, softmax + write +0.25.
// Microkernel (csrc/router_cc_bench.cu): all 384 logits at 2.05 us (rows 1) / 2.6 (rows 2) with
// 20-30 warps x 2-3 chunks per lane; 8 warps x 12 chunks 2.9; the issue is back-pressured by the SM's
// outstanding-request capacity (2 loads/lane still take 1.5 us to issue); cp.async.bulk per warp
// 2.75, one 30 KB bulk per CTA 3.4; contiguous vs round-robin expert rows: no difference.
// Equality vs the 96 x 4 WMMA path (tests/test_frontend_fe78.py --mma cc): 3000 + 1200 row-evaluations,
// 0 top-8 index-set mismatches, 2 bf16 rounding flips, x/x_sf bit-identical; 8-rank
// test_four_api_correctness (fused mxfp4 + qoq, tokens/rank 1, 2, 8): cos_min 0.99999 / 0.99993.
// Router CTA count the launch will use (bench / stamp attribution helper).
// `k_parts` (DG_FE_TINYM_KPARTS, full-K grid only): 1 | 2 | 4 K-parts per expert
// group (grid=97,k_parts=4 = the legacy 24 x 16 x 4 layout inside the full-K
// framework; grid=78,k_parts=2 = 35 groups x 11 experts x 2 = 70 + 1 CTAs).
int router_quant_topk_frontend_router_ctas(int m, int h, int e, int topk, int tiny, int grid, int k_parts);
void launch_router_quant_topk_frontend(
    const void* hidden, const void* router_weight,
    void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
    void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
    int tiny, int stamps_on, int l2_persist, int pdl_mode, int grid, int mma, int k_parts, cudaStream_t stream);
