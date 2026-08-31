#include "mega_frontend.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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


namespace {

constexpr int GENERIC_MAX_EXPERTS = 384;
constexpr int GENERIC_WARPS = 8;
constexpr int GENERIC_THREADS = GENERIC_WARPS * 32;

__device__ __forceinline__ float warp_max_reduce(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ float warp_sum_reduce(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ uint16_t generic_cvt_e4m3x2(float x0, float x1) {
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
        : "=h"(out) : "f"(x0), "f"(x1));
    return out;
}

template <bool QOQ>
__global__ void generic_router_quant_topk_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ router_weight,
    void* __restrict__ x_storage,
    float* __restrict__ x_sf,
    int64_t* __restrict__ topk_idx,
    float* __restrict__ topk_weights,
    int h, int e, int topk) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ float logits[GENERIC_MAX_EXPERTS];
    __shared__ float warp_amax[GENERIC_WARPS];
    __shared__ float row_scale;

    const __nv_bfloat16* xrow = hidden + static_cast<int64_t>(token) * h;

    // Eight warps cooperatively compute all router logits. Each warp owns one
    // expert at a time and reduces the H dot product in registers.
    for (int expert = warp; expert < e; expert += GENERIC_WARPS) {
        const __nv_bfloat16* wrow =
            router_weight + static_cast<int64_t>(expert) * h;
        float accum = 0.0f;
        for (int kk = lane; kk < h; kk += 32)
            accum = fmaf(__bfloat162float(xrow[kk]),
                         __bfloat162float(wrow[kk]), accum);
        accum = warp_sum_reduce(accum);
        if (lane == 0) logits[expert] = accum;
    }
    __syncthreads();

    if constexpr (QOQ) {
        float local_max = 0.0f;
        for (int kk = tid; kk < h; kk += GENERIC_THREADS)
            local_max = fmaxf(local_max, fabsf(__bfloat162float(xrow[kk])));
        local_max = warp_max_reduce(local_max);
        if (lane == 0) warp_amax[warp] = local_max;
        __syncthreads();
        if (warp == 0) {
            float value = lane < GENERIC_WARPS ? warp_amax[lane] : 0.0f;
            value = warp_max_reduce(value);
            if (lane == 0)
                row_scale = fmaxf(value * (1.0f / 127.0f), 1.0e-30f);
        }
        __syncthreads();
        const float inv = 1.0f / row_scale;
        int8_t* qrow = static_cast<int8_t*>(x_storage) +
                       static_cast<int64_t>(token) * h;
        for (int kk = tid; kk < h; kk += GENERIC_THREADS) {
            int q = __float2int_rn(__bfloat162float(xrow[kk]) * inv);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            qrow[kk] = static_cast<int8_t>(q);
        }
        for (int g = tid; g < h / 128; g += GENERIC_THREADS)
            x_sf[static_cast<int64_t>(token) * (h / 128) + g] = row_scale;
    } else {
        uint8_t* qrow = static_cast<uint8_t*>(x_storage) +
                        static_cast<int64_t>(token) * h;
        // One warp owns one K128 activation block. Amax, scale and conversion
        // remain warp-local and require no block-wide reduction.
        for (int group = warp; group < h / 128; group += GENERIC_WARPS) {
            float values[4];
            float local_max = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int kk = group * 128 + lane * 4 + j;
                values[j] = __bfloat162float(xrow[kk]);
                local_max = fmaxf(local_max, fabsf(values[j]));
            }
            local_max = warp_max_reduce(local_max);
            const float scale = __shfl_sync(
                0xffffffffu,
                fmaxf(local_max * (1.0f / 448.0f), 1.0e-30f), 0);
            if (lane == 0)
                x_sf[static_cast<int64_t>(token) * (h / 128) + group] = scale;
            uint16_t* out = reinterpret_cast<uint16_t*>(
                qrow + group * 128 + lane * 4);
            out[0] = generic_cvt_e4m3x2(values[0] / scale, values[1] / scale);
            out[1] = generic_cvt_e4m3x2(values[2] / scale, values[3] / scale);
        }
    }

    // Deterministic top-k and softmax. The customer balanced benchmark
    // overwrites these IDs after the frontend, but this preserves full API
    // semantics for the new H3072/E384/TopK8 shape.
    if (tid == 0) {
        float best_value[8];
        int best_id[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            best_value[j] = -3.402823466e+38F;
            best_id[j] = -1;
        }
        for (int expert = 0; expert < e; ++expert) {
            const float value = logits[expert];
            int pos = topk;
            for (int j = 0; j < topk; ++j) {
                if (value > best_value[j]) { pos = j; break; }
            }
            if (pos < topk) {
                for (int j = topk - 1; j > pos; --j) {
                    best_value[j] = best_value[j - 1];
                    best_id[j] = best_id[j - 1];
                }
                best_value[pos] = value;
                best_id[pos] = expert;
            }
        }
        float max_value = best_value[0];
        float sum = 0.0f;
        for (int j = 0; j < topk; ++j) {
            best_value[j] = __expf(best_value[j] - max_value);
            sum += best_value[j];
        }
        for (int j = 0; j < topk; ++j) {
            topk_idx[static_cast<int64_t>(token) * topk + j] = best_id[j];
            topk_weights[static_cast<int64_t>(token) * topk + j] =
                best_value[j] / sum;
        }
    }
}


__global__ void mxfp4_quant_topk_from_logits_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ logits,
    uint8_t* __restrict__ x_fp8,
    float* __restrict__ x_sf,
    int64_t* __restrict__ topk_idx,
    float* __restrict__ topk_weights,
    int h, int e, int topk) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const __nv_bfloat16* xrow = hidden + static_cast<int64_t>(token) * h;
    uint8_t* qrow = x_fp8 + static_cast<int64_t>(token) * h;

    for (int group = warp; group < h / 128; group += GENERIC_WARPS) {
        float values[4];
        float local_max = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int kk = group * 128 + lane * 4 + j;
            values[j] = __bfloat162float(xrow[kk]);
            local_max = fmaxf(local_max, fabsf(values[j]));
        }
        local_max = warp_max_reduce(local_max);
        const float scale = __shfl_sync(
            0xffffffffu, fmaxf(local_max * (1.0f / 448.0f), 1.0e-30f), 0);
        if (lane == 0)
            x_sf[static_cast<int64_t>(token) * (h / 128) + group] = scale;
        uint16_t* out = reinterpret_cast<uint16_t*>(qrow + group * 128 + lane * 4);
        out[0] = generic_cvt_e4m3x2(values[0] / scale, values[1] / scale);
        out[1] = generic_cvt_e4m3x2(values[2] / scale, values[3] / scale);
    }

    if (tid == 0) {
        float best_value[8];
        int best_id[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) { best_value[j] = -3.402823466e+38F; best_id[j] = -1; }
        const __nv_bfloat16* lrow = logits + static_cast<int64_t>(token) * e;
        for (int expert = 0; expert < e; ++expert) {
            float value = __bfloat162float(lrow[expert]);
            int pos = topk;
            for (int j = 0; j < topk; ++j) if (value > best_value[j]) { pos = j; break; }
            if (pos < topk) {
                for (int j = topk - 1; j > pos; --j) {
                    best_value[j] = best_value[j - 1]; best_id[j] = best_id[j - 1];
                }
                best_value[pos] = value; best_id[pos] = expert;
            }
        }
        float mx = best_value[0], sum = 0.0f;
        for (int j = 0; j < topk; ++j) { best_value[j] = __expf(best_value[j] - mx); sum += best_value[j]; }
        for (int j = 0; j < topk; ++j) {
            topk_idx[static_cast<int64_t>(token) * topk + j] = best_id[j];
            topk_weights[static_cast<int64_t>(token) * topk + j] = best_value[j] / sum;
        }
    }
}


} // namespace

void launch_mxfp4_quant_topk_from_logits(
    const void* hidden, const void* logits, void* x_fp8, void* x_sf,
    void* topk_idx, void* topk_weights, int m, int h, int e, int topk,
    cudaStream_t stream) {
    mxfp4_quant_topk_from_logits_kernel<<<m, GENERIC_THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(logits), static_cast<uint8_t*>(x_fp8),
        static_cast<float*>(x_sf), static_cast<int64_t*>(topk_idx),
        static_cast<float*>(topk_weights), h, e, topk);
}


void launch_mxfp4_router_quant_topk_generic(
    const void* hidden, const void* router_weight, void* x_fp8, void* x_sf,
    void* topk_idx, void* topk_weights, int m, int h, int e, int topk,
    cudaStream_t stream) {
    generic_router_quant_topk_kernel<false><<<m, GENERIC_THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(router_weight), x_fp8,
        static_cast<float*>(x_sf), static_cast<int64_t*>(topk_idx),
        static_cast<float*>(topk_weights), h, e, topk);
}

void launch_qoq_router_quant_topk_generic(
    const void* hidden, const void* router_weight, void* x_int8, void* x_sf,
    void* topk_idx, void* topk_weights, int m, int h, int e, int topk,
    cudaStream_t stream) {
    generic_router_quant_topk_kernel<true><<<m, GENERIC_THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(router_weight), x_int8,
        static_cast<float*>(x_sf), static_cast<int64_t*>(topk_idx),
        static_cast<float*>(topk_weights), h, e, topk);
}
