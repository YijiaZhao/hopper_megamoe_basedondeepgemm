#pragma once
// Fable cc-router top-8 selection + softmax over a compact key array, shared by the
// standalone frontend (csrc/fable_frontend.cu, last-arriver merge) and the SM90 fused
// MegaMoE kernel (DG_FE_SELECT_IN_MEGA=1: the FE ends after the router CTAs stored their
// keys; the Mega prologue's idle warps select while warps 0..2 initialise smem/barriers).
// Key = orderable(bf16 logit) << 16 | (0xFFFF - expert): u32 order == (value desc, id asc).
// Token t's e (= 384) keys are contiguous: lane l reads 16 B vectors l, l + 32, l + 64
// (12 keys), insertion-sorts them into a per-lane top-8, merge8 fuses the 32 lists with
// 8 redux rounds, softmax over the 8 selected bf16 logits in the legacy k = 0..7 order.
#include <cstdint>
#include <cuda_bf16.h>

namespace fable_cc {
constexpr int kTopK8 = 8;
__device__ __forceinline__ float topk_key_value(uint32_t key) {
    const uint32_t o = key >> 16;
    const uint32_t b = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
    return __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(b)));
}
__device__ __forceinline__ int topk_key_index(uint32_t key) {
    return static_cast<int>(0xFFFFu - (key & 0xFFFFu));
}
__device__ __forceinline__ uint32_t warp_max_u32_redux(uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_max_sync(0xffffffffu, v);
#else
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
#endif
}
__device__ __forceinline__ float warp_max_f32(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint4 ld_cg_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
// 8 redux rounds: `run` (one key per lane, from a previous batch; 0 = none) and every lane's
// sorted list `loc` (desc) -> lanes 0..7 of `run` hold the top-8 (desc), other lanes 0.
template <int kTopK>
__device__ __forceinline__ void merge8(uint32_t& run, uint32_t (&loc)[kTopK], int lane) {
    uint32_t out = 0u;
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const uint32_t best = warp_max_u32_redux(max(run, loc[0]));
        if (lane == k) out = best;
        if (run == best) run = 0u;
        if (loc[0] == best) {                 // pop this lane's head
            #pragma unroll
            for (int j = 0; j < kTopK - 1; ++j) loc[j] = loc[j + 1];
            loc[kTopK - 1] = 0u;
        }
    }
    run = lane < kTopK ? out : 0u;
}
// softmax over the 8 selected bf16 logits (lanes 0..7 of run), legacy order (k = 0..7 sequential sum)
template <int kTopK>
__device__ __forceinline__ void topk_finish(uint32_t run, int lane, int t, int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights) {
    const float sel_v = lane < kTopK ? topk_key_value(run) : -INFINITY;
    const float mx = warp_max_f32(sel_v);
    const float ex = lane < kTopK ? expf(sel_v - mx) : 0.0f;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) sum += __shfl_sync(0xffffffffu, ex, k);
    if (lane < kTopK) {
        topk_idx[static_cast<int64_t>(t) * kTopK + lane] = topk_key_index(run);
        topk_weights[static_cast<int64_t>(t) * kTopK + lane] = ex / sum;
    }
}
// Insert the lane's kVec x 4 keys into a sorted (desc) per-lane top-kTopK list.
template <int kTopK, int kVec>
__device__ __forceinline__ void insert_keys(const uint4 (&q)[kVec], uint32_t (&loc)[kTopK]) {
    #pragma unroll
    for (int j = 0; j < kTopK; ++j) loc[j] = 0u;
    #pragma unroll
    for (int i = 0; i < kVec; ++i) {
        const uint32_t kv[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            uint32_t k = kv[u];
            #pragma unroll
            for (int j = 0; j < kTopK; ++j) { const uint32_t lo = min(k, loc[j]); loc[j] = max(k, loc[j]); k = lo; }
        }
    }
}
// One warp, token t: keys_t = the token's 384 contiguous keys (complete: the producer kernel
// has finished); writes topk_idx/topk_weights[t][0..7].
__device__ __forceinline__ void select_topk8_compact384(const uint32_t* __restrict__ keys_t, int lane, int t,
                                                        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights) {
    constexpr int kVec = 3;
    uint4 q[kVec];
    #pragma unroll
    for (int i = 0; i < kVec; ++i) q[i] = ld_cg_v4(keys_t + 4 * (lane + 32 * i));
    uint32_t run = 0u;
    uint32_t loc[kTopK8];
    insert_keys<kTopK8, kVec>(q, loc);
    merge8<kTopK8>(run, loc, lane);
    topk_finish<kTopK8>(run, lane, t, topk_idx, topk_weights);
}
}  // namespace fable_cc
