#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Fused MegaMoE frontend for SM90: router logits (bf16 in, fp32 accumulate,
// bf16-rounded) + deterministic top-k + softmax over the selected logits +
// online activation quantization, written straight into the MegaMoE symmetric
// input views. `mode`: 0 = FP8 E4M3 per-token/K128 (NVFP4/MXFP4 weights),
// 1 = INT8 per-token whole-row scale repeated over the K128 slots (QoQ).
// `workspace` layout: [0,256) counters, [256, 256 + 4*64*e*4) K-split partial
// logits, 64*e*4 bytes of top-k keys (tiny-M), then kFrontendStampsBytes of
// optional per-CTA phase stamps (DG_FE_STAMPS=1). Zero-initialised once.
// `tiny` (DG_FE_TINYM): m <= 16 single-wave config (3 smem stages) -- same
// math, bit-identical outputs. `stamps_on`: record %globaltimer phase stamps.
constexpr size_t kFrontendStampsOffsetBase = 256;
constexpr size_t kFrontendMaxCTAs = 256;
constexpr size_t kFrontendStampsBytes = kFrontendMaxCTAs * 8 * sizeof(unsigned long long);
size_t router_quant_topk_frontend_keys_offset(int e);     // [m, e] u32 bf16|id keys (tiny-M)
size_t router_quant_topk_frontend_stamps_offset(int e);
size_t router_quant_topk_frontend_workspace_bytes(int e);
void launch_router_quant_topk_frontend(
    const void* hidden, const void* router_weight,
    void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
    void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
    int tiny, int stamps_on, cudaStream_t stream);
