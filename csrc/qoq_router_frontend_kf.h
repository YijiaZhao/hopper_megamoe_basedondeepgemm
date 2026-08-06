#pragma once
#include <cuda_runtime.h>
void qoq_router_frontend_launcher(
    const void* hidden, const void* router_weight, void* x_int8, void* x_sf,
    void* topk_weights, void* topk_ids, int m, cudaStream_t stream);
