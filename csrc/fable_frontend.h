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
// `grid` (DG_FE_TINYM_GRID): 96 = legacy tiny-M split (24 expert groups x 4
// K-parts + m quant/top-k CTAs); 0 = auto: full-K scheme sized to the SM count
// (H20: 77 router CTAs x 5 experts + 1 merger CTA = 78); N > 0 = full-K scheme
// with N CTAs in total. Full-K outputs are deterministic but not bit-identical to
// the legacy split (different fp32 accumulation order before the bf16 rounding).
size_t router_quant_topk_frontend_workspace_bytes(int e);
// Router CTA count the launch will use (bench / stamp attribution helper).
int router_quant_topk_frontend_router_ctas(int m, int h, int e, int topk, int tiny, int grid);
void launch_router_quant_topk_frontend(
    const void* hidden, const void* router_weight,
    void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
    void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
    int tiny, int stamps_on, int l2_persist, int pdl_mode, int grid, cudaStream_t stream);
