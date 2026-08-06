#include "mega_frontend.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__global__ void qoq_quant_topk_kernel(const __nv_bfloat16* __restrict__ hidden,
                                      const __nv_bfloat16* __restrict__ logits,
                                      int8_t* __restrict__ xq,
                                      float* __restrict__ xsf,
                                      int64_t* __restrict__ topk_idx,
                                      float* __restrict__ topk_weights,
                                      int h, int e, int topk) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_max[8];
    __shared__ float row_scale;
    float amax = 0.f;
    const auto* xrow = hidden + (int64_t)token * h;
    for (int k = tid; k < h; k += blockDim.x)
        amax = fmaxf(amax, fabsf(__bfloat162float(xrow[k])));
    for (int d = 16; d; d >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, d));
    if ((tid & 31) == 0) warp_max[tid >> 5] = amax;
    __syncthreads();
    if (tid < 32) {
        float v = tid < 8 ? warp_max[tid] : 0.f;
        for (int d = 16; d; d >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
        if (tid == 0) row_scale = fmaxf(v * (1.f / 127.f), 1.e-30f);
    }
    __syncthreads();
    const float inv = 1.f / row_scale;
    auto* qrow = xq + (int64_t)token * h;
    for (int k = tid; k < h; k += blockDim.x) {
        int q = __float2int_rn(__bfloat162float(xrow[k]) * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qrow[k] = (int8_t)q;
    }
    if (tid < h / 128) xsf[(int64_t)token * (h / 128) + tid] = row_scale;

    if (tid == 0) {
        float vals[8]; int ids[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) { vals[j] = -3.402823466e+38F; ids[j] = -1; }
        const auto* lrow = logits + (int64_t)token * e;
        for (int i = 0; i < e; ++i) {
            float v = __bfloat162float(lrow[i]);
            int p = topk;
            for (int j = 0; j < topk; ++j) if (v > vals[j]) { p = j; break; }
            if (p < topk) {
                for (int j = topk - 1; j > p; --j) { vals[j] = vals[j-1]; ids[j] = ids[j-1]; }
                vals[p] = v; ids[p] = i;
            }
        }
        float sum = 0.f, mx = vals[0];
        for (int j = 0; j < topk; ++j) { vals[j] = __expf(vals[j] - mx); sum += vals[j]; }
        for (int j = 0; j < topk; ++j) {
            topk_idx[(int64_t)token * topk + j] = ids[j];
            topk_weights[(int64_t)token * topk + j] = vals[j] / sum;
        }
    }
}

void launch_qoq_quant_topk(const void* hidden, const void* logits, void* x_int8,
                           void* x_sf, void* topk_idx, void* topk_weights,
                           int m, int h, int e, int topk, cudaStream_t stream) {
    qoq_quant_topk_kernel<<<m, 256, 0, stream>>>(
        (const __nv_bfloat16*)hidden, (const __nv_bfloat16*)logits,
        (int8_t*)x_int8, (float*)x_sf, (int64_t*)topk_idx, (float*)topk_weights,
        h, e, topk);
}
