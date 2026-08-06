#pragma once
#include <cuda_runtime.h>
void launch_qoq_quant_topk(const void* hidden, const void* logits, void* x_int8,
                           void* x_sf, void* topk_idx, void* topk_weights,
                           int m, int h, int e, int topk, cudaStream_t stream);
