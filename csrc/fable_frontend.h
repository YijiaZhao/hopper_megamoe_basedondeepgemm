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
// H20-3e (.7) standalone, rows 1|2 x mxfp4|qoq, kernel-end stamp median (us):
//   96 x 4 wmma 6.91 (all 4 cells) | 96 x 4 swapab row 6.66 (-0.25) | 96 x 4 swapab fragment
//   6.14 (-0.77) | 78 full-K wmma 7.94-8.45 | 78 full-K swapab 6.40-6.91.
//   NCU (96, rows 1, mxfp4): L2 read requests 49.1k (wmma) -> 37.7k (swapab row), read sectors
//   113k -> 77k, LSU instructions 89k -> 29k; fragment layout: see README knob row.
// Default stays 0 (row-major weights): swapab alone is < 0.3 us; the fragment variant needs the
// caller to permute the router weight at weight-transform time (opt-in).
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
