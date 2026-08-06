#pragma once

#include <cuda_runtime.h>

void router_frontend_launcher(
    const void* hidden,
    const void* router_weight,
    void* x_fp8,
    void* x_sf,
    void* topk_weights,
    void* topk_ids,
    int m,
    cudaStream_t stream);
