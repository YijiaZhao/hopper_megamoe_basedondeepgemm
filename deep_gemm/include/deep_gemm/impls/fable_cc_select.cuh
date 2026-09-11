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
// DG_FE_CC_SELECT=pruned in the fused Mega: the JIT host injects `#define DG_FE_CC_SELECT_PRUNED 1`
#ifndef DG_FE_CC_SELECT_PRUNED
#define DG_FE_CC_SELECT_PRUNED 0
#endif

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
// ---------------------------------------------------------------- pruned select
// DG_FE_CC_SELECT=pruned (round 5): two-level threshold instead of the 12 x 8 insertion + 8 merge
// rounds. Level 1: lm = every lane's max over its 12 keys; the 32 lane maxes are ranked by an
// all-gather (32 shfl.idx + compares: independent instructions, no serial redux chain);
// bound = the lane max of rank 7 (8th largest). Eight distinct keys are >= bound, so the global
// top-8 is a subset of the keys >= bound. Level 2: if no lane holds a SECOND key >= bound
// (one ballot), the top-8 IS the ranked lane maxes: lane r < 8 fetches the lane max of rank r
// (fast path: zero redux rounds). Otherwise every lane keeps its keys >= bound as <= 3
// candidates (max, 2nd max, 3rd max) and 8 redux rounds select over them; a lane with >= 4
// keys >= bound (ballot, rare) falls back to insertion + merge8. Keys are unique, so the result
// (8 keys, descending, lanes 0..7 of `run`) is identical to insertion + merge8 on every path.
__device__ __forceinline__ uint32_t max12(const uint4 (&q)[3]) {
    uint32_t m = max(max(q[0].x, q[0].y), max(q[0].z, q[0].w));
    m = max(m, max(max(q[1].x, q[1].y), max(q[1].z, q[1].w)));
    return max(m, max(max(q[2].x, q[2].y), max(q[2].z, q[2].w)));
}
// max over the lane's 12 keys excluding the values a and b (0 = none left)
__device__ __forceinline__ uint32_t max12_excluding(const uint4 (&q)[3], uint32_t a, uint32_t b) {
    uint32_t m = 0u;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const uint32_t kv[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) m = max(m, (kv[u] == a || kv[u] == b) ? 0u : kv[u]);
    }
    return m;
}
__device__ __forceinline__ int count12_ge(const uint4 (&q)[3], uint32_t bound) {
    int c = 0;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const uint32_t kv[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) c += kv[u] >= bound ? 1 : 0;
    }
    return c;
}
template <int kTopK>
__device__ __forceinline__ void select_pruned(const uint4 (&q)[3], int lane, uint32_t& run) {
    static_assert(kTopK == 8, "pruned select assumes top-8 over 12 keys per lane");
    const uint32_t lm = max12(q);
    int rank = 0;                                             // # lane maxes above mine (unique keys -> a permutation)
    #pragma unroll
    for (int j = 0; j < 32; ++j) rank += __shfl_sync(0xffffffffu, lm, j) > lm ? 1 : 0;
    const uint32_t bound = __shfl_sync(0xffffffffu, lm, __ffs(__ballot_sync(0xffffffffu, rank == kTopK - 1)) - 1);
    const int cnt = count12_ge(q, bound);                     // 0 for rank >= 8 lanes, >= 1 for the 8 others
    if (__ballot_sync(0xffffffffu, cnt > 1) == 0u) {          // fast path: the top-8 are the 8 largest lane maxes
        int src = 0;
        #pragma unroll
        for (int r = 0; r < kTopK; ++r) {
            const uint32_t bm = __ballot_sync(0xffffffffu, rank == r);
            if (lane == r) src = __ffs(bm) - 1;
        }
        const uint32_t v = __shfl_sync(0xffffffffu, lm, src);
        run = lane < kTopK ? v : 0u;
        return;
    }
    if (__ballot_sync(0xffffffffu, cnt > 3) != 0u) {          // rare: >= 4 keys >= bound in one lane -> full path
        uint32_t loc[kTopK];
        insert_keys<kTopK, 3>(q, loc);
        run = 0u;
        merge8<kTopK>(run, loc, lane);
        return;
    }
    const uint32_t m2 = max12_excluding(q, lm, 0u), m3 = max12_excluding(q, lm, m2);
    uint32_t c0 = lm >= bound ? lm : 0u, c1 = m2 >= bound ? m2 : 0u, c2 = m3 >= bound ? m3 : 0u;
    uint32_t out = 0u;
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const uint32_t best = warp_max_u32_redux(max(c0, max(c1, c2)));
        if (lane == k) out = best;
        c0 = c0 == best ? 0u : c0;
        c1 = c1 == best ? 0u : c1;
        c2 = c2 == best ? 0u : c2;
    }
    run = lane < kTopK ? out : 0u;
}
// One warp, token t: keys_t = the token's 384 contiguous keys (complete: the producer kernel
// has finished, or the caller verified them); writes topk_idx/topk_weights[t][0..7].
// pruned (DG_FE_CC_SELECT=pruned): select_pruned above instead of insertion + merge8 (same result).
__device__ __forceinline__ void select_topk8_compact384(const uint32_t* __restrict__ keys_t, int lane, int t,
                                                        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
                                                        bool pruned = false) {
    constexpr int kVec = 3;
    uint4 q[kVec];
    #pragma unroll
    for (int i = 0; i < kVec; ++i) q[i] = ld_cg_v4(keys_t + 4 * (lane + 32 * i));
    uint32_t run = 0u;
    if (pruned) {
        select_pruned<kTopK8>(q, lane, run);
    } else {
        uint32_t loc[kTopK8];
        insert_keys<kTopK8, kVec>(q, loc);
        merge8<kTopK8>(run, loc, lane);
    }
    topk_finish<kTopK8>(run, lane, t, topk_idx, topk_weights);
}
}  // namespace fable_cc
