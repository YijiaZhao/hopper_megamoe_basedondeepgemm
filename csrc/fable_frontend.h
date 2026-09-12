#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Fused MegaMoE frontend for SM90: router logits (bf16 in, fp32 accumulate,
// bf16-rounded) + deterministic top-k + softmax over the selected logits +
// online activation quantization, written straight into the MegaMoE symmetric
// input views. `mode`: 0 = FP8 E4M3 per-token/K128 (NVFP4/MXFP4 weights),
// 1 = INT8 per-token whole-row scale repeated over the K128 slots (QoQ).
// `workspace` layout: [0,256) tickets, [256, 256 + 4*64*e*4) K-split partial
// logits (the cc router's compact [token][e] u32 key array sits at +64 KB inside
// it), then kFrontendStampsBytes of optional per-CTA phase stamps (DG_FE_STAMPS=1).
// Zero-initialised once.
constexpr size_t kFrontendStampsOffsetBase = 256;
constexpr size_t kFrontendMaxCTAs = 256;
constexpr size_t kFrontendStampsBytes = kFrontendMaxCTAs * 8 * sizeof(unsigned long long);

// Launch shape chosen from (m, h, e, topk) alone (`select_fe_path`; the python wrapper
// mirrors it to decide the router-weight layout):
//   kFEPathCC     m <= 2 rows, h = 3072, e = 384, top-8: `router_cc_lean_kernel`, the CUDA-core
//                 K-split router (77 CTAs x 5 experts x 4 K-part warps, weights straight into
//                 registers, compact 384-key array per token, relaxed atomic ticket, the last
//                 router CTA selects top-8 + softmax; `select_in_mega` = 1 stops after the keys
//                 and the fused MegaMoE prologue selects, deep_gemm/impls/fable_cc_select.cuh);
//                 the spare CTA quantises the rows. Router weights sit in the persisting L2
//                 set-aside when `l2_persist` = 1 (DG_FE_ROUTER_L2_PERSIST, the wrapper's
//                 default on this path).
//   kFEPathSwapAB m <= 16, h = 3072, top-8: legacy 96 x 4 grid (24 expert groups x 4 K-parts +
//                 m quant / top-k CTAs), experts on the MMA M dimension (mma.sync m16n8k16
//                 bf16 -> fp32), weight A fragments ld.global.nc straight into registers;
//                 `wlayout` 1 = the router weights were permuted once on the host into
//                 A-fragment order (fable_router_weight_fragment_layout: one warp load = one
//                 contiguous 512 B run), 0 = row-major [e][h].
//   kFEPathWMMA   every other shape: 96-CTA (m <= 16, single-wave 3-stage) or 48-CTA grid,
//                 cp.async smem ring + WMMA bf16 m16n16k16, m quant / top-k CTAs.
// All three are deterministic; the cc and swapab paths are not bit-identical to WMMA
// (different fp32 accumulation order before the bf16 logit rounding).
constexpr int kFEPathWMMA = 0;
constexpr int kFEPathSwapAB = 1;
constexpr int kFEPathCC = 2;
int select_fe_path(int m, int h, int e, int topk);
size_t router_quant_topk_frontend_workspace_bytes(int e);
// Router CTA count the launch will use (bench / stamp attribution helper).
int router_quant_topk_frontend_router_ctas(int m, int h, int e, int topk);
void launch_router_quant_topk_frontend(
    const void* hidden, const void* router_weight,
    void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
    void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
    int stamps_on, int l2_persist, int wlayout, int select_in_mega, cudaStream_t stream);
