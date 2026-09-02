#pragma once
#include <cuda_runtime.h>

// Shape-generic tensor-core fused frontend: bf16 router GEMM (WMMA, fp32 accumulate,
// bf16-rounded logits) + deterministic top-k (largest logit, smallest id on ties) +
// softmax over the selected logits + online activation quantization.
//   mode 0: FP8 E4M3 per token / K128 (scaled MXFP4 / NVFP4 weights)
//   mode 1: INT8 per token, one scale repeated over the K128 slots (QoQ W4A8)
// Requirements: 1 <= m <= 64, h % 1024 == 0, e % 16 == 0, e <= 512, 1 <= topk <= 8.
void launch_router_quant_topk_tc(
    const void* hidden, const void* router_weight,
    void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
    int m, int h, int e, int topk, int mode, cudaStream_t stream);
bool router_quant_topk_tc_supported(int m, int h, int e, int topk);
