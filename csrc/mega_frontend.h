#pragma once
#include <cuda_runtime.h>
void launch_qoq_quant_topk(const void* hidden, const void* logits, void* x_int8,
                           void* x_sf, void* topk_idx, void* topk_weights,
                           int m, int h, int e, int topk, cudaStream_t stream);

void launch_mxfp4_router_quant_topk_generic(
    const void* hidden, const void* router_weight, void* x_fp8, void* x_sf,
    void* topk_idx, void* topk_weights, int m, int h, int e, int topk,
    cudaStream_t stream);
void launch_qoq_router_quant_topk_generic(
    const void* hidden, const void* router_weight, void* x_int8, void* x_sf,
    void* topk_idx, void* topk_weights, int m, int h, int e, int topk,
    cudaStream_t stream);
void launch_mxfp4_quant_topk_from_logits(
    const void* hidden, const void* logits, void* x_fp8, void* x_sf,
    void* topk_idx, void* topk_weights, int m, int h, int e, int topk,
    cudaStream_t stream);
