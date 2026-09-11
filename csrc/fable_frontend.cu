// Fused router + top-k + activation-quantization frontend for the SM90 fused
// MegaMoE (see mega_frontend.h). One launch, two CTA roles:
//   * router CTAs (E / 8 of them): warp w owns expert 8*blockIdx + w, streams
//     its weight row once from HBM and accumulates the dot product with every
//     token (hidden rows are L2-resident). Logits are rounded to bf16 (matching
//     a bf16 router matmul) and parked in the workspace; the last router CTA to
//     finish (atomic ticket) runs top-k + softmax for all tokens.
//   * quant CTAs (m of them): one token each, per-K128 FP8 (mode 0) or whole
//     row INT8 (mode 1), scale computed online.
// Tiny-M full-K scheme (kFullK, DG_FE_TINYM_GRID=auto|N, default for m <= 16):
// see the "full-K router" section below -- router CTA count = SM count - 1,
// every CTA owns ~5 experts over the whole K, final logits per CTA, streaming
// top-8 merge in one merger CTA, quantisation on the router CTAs' idle time.
#include "fable_frontend.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

namespace {

constexpr int kExpertsPerCTA = 16;
constexpr int kThreads = 256;
constexpr int kMaxTopK = 8;
constexpr int kMaxExperts = 512;
// full-K scheme limits
constexpr int kFullKStages = 12;                 // K chunks in flight per CTA (h <= 3072 -> all of them)
constexpr int kCandSlots = 8;                    // candidate keys per (token, router CTA) -> experts_per_cta <= 8
constexpr int kMaxRouterCTAs = 128;
constexpr int kMaxSlotsPerLane = kMaxRouterCTAs * kCandSlots / 32;   // merger warp: keys per lane

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float half_warp_max(float v) {
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t cvt_e4m3x2(float x0, float x1) {
    uint16_t out = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(out) : "f"(x0), "f"(x1));
#endif
    return out;
}
__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// Optional per-CTA phase stamps (DG_FE_STAMPS=1): [blockIdx][8] u64 ns.
// router CTA: 0 start / 1 chunk0 landed / 2 mma done / 3 ticket bumped
// quant  CTA: 0 start / 1 quant done / 2 ticket seen / 3 top-k done
//             (tiny top-k: 4 partial logits loaded / 5 selection rounds done)
// full-K router CTA: 0 start / 1 chunk0 landed / 2 mma done / 3 keys written /
//                    4 quant done (CTAs t < m only) / 5 all chunks issued /
//                    6 %smid / 7 prologue done (before issue)
// full-K merger CTA (token 0's warp): 0 start / 1 first CTA seen / 2 last CTA
//                    seen / 3 merge done / 4 top-k written
constexpr int kStampSlots = 8;
__device__ __forceinline__ void stamp(unsigned long long* stamps, int slot) {
    if (stamps != nullptr && threadIdx.x == 0) stamps[blockIdx.x * kStampSlots + slot] = globaltimer_ns();
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 t = __bfloat1622float2(p[j]);
        f[2 * j] = t.x; f[2 * j + 1] = t.y;
    }
}

// ---------------------------------------------------------------- router CTA
// Tensor-core router GEMM: a CTA owns kExpertsPerCTA (= 16) experts and all m
// tokens. Hidden [m x 256] and weight [16 x 256] K-chunks are staged in smem;
// warp w computes the (m-tile, K-slice) product with WMMA bf16 m16n16k16 in
// fp32; K-slice partials are reduced through smem at the end. Hidden rows are
// read once per CTA (L2-resident), weights once overall.
constexpr int kChunkK = 256;
constexpr int kSmemLd = kChunkK + 8;     // bf16 elements per smem row (bank-spread)
constexpr int kMaxMTiles = 4;            // m <= 64

// Pipeline depth per m-tile count: fewer token rows -> smaller stages -> deeper
// pipeline (the loop is HBM/L2 latency bound, compute is negligible).
// kTiny (DG_FE_TINYM, m <= 16 only): 3 stages instead of 8. With 8 stages the
// dynamic smem is 143 KB -> 1 CTA/SM, so the 96 router CTAs + m quant CTAs do
// not fit on the 78 SMs of an H20 and run as two waves (the quant/top-k CTAs
// are in the second wave). 3 stages = 59 KB -> 3 CTAs/SM, single wave. H=3072
// needs exactly 3 chunks per K-part, so nothing is pipelined away. The math
// (WMMA order, K-split partial order, bf16 rounding) is unchanged -> bit-identical.
template <int kMTiles, bool kTiny = false> struct RouterCfg {
    static_assert(!kTiny || kMTiles == 1, "tiny-M config is the m <= 16 path");
    // K-split across CTAs: partial logits go to workspace slice [ks][m][e] and are
    // summed in fixed order by the top-k CTA (deterministic).
    static constexpr int kKSplitCTAs = kMTiles == 1 ? 4 : 2;
    static constexpr int kStages = kTiny ? 3 : (kMTiles == 1 ? 8 : (kMTiles == 2 ? 6 : 4));
    static constexpr int kHRows = kMTiles * 16;
    static constexpr int kStageElems = (kHRows + kExpertsPerCTA) * kSmemLd;   // bf16 per stage
    static constexpr int kDynSmemBytes = kStages * kStageElems * 2 + (kThreads / 32) * 256 * 4;
};

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
// DG_FE_ROUTER_L2_PERSIST=2: the router-weight cp.async carry an L2::evict_last
// cache-policy hint so the 2.4 MB router matrix outranks the (evict_normal) MoE
// weights that stream through L2 between launches. No host-side set-aside.
__device__ __forceinline__ uint64_t l2_evict_last_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
__device__ __forceinline__ void cp_async_16_hint(void* smem_dst, const void* gmem_src, uint64_t policy) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(gmem_src), "l"(policy) : "memory");
}

template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_role(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        float* __restrict__ logits,          // [m, e] workspace
        unsigned long long* stamps,
        int m, int h, int e, int w_hint) {
    using namespace nvcuda;
    using Cfg = RouterCfg<kMTiles, kTiny>;
    constexpr int kStages = Cfg::kStages;
    constexpr int kHRows = Cfg::kHRows;
    constexpr int kStageElems = Cfg::kStageElems;
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    auto stage_h = [&](int st) {
        return reinterpret_cast<__nv_bfloat16 (*)[kSmemLd]>(dyn_smem + st * kStageElems * 2);
    };
    auto stage_w = [&](int st) {
        return reinterpret_cast<__nv_bfloat16 (*)[kSmemLd]>(dyn_smem + (st * kStageElems + kHRows * kSmemLd) * 2);
    };
    float (*part_s)[256] = reinterpret_cast<float (*)[256]>(dyn_smem + kStages * kStageElems * 2);
    constexpr int kKSplit = (kThreads / 32) / kMTiles;          // warps per m-tile
    constexpr int kKPerWarp = kChunkK / kKSplit;                // 256 / {8,4,2}
    constexpr int kVecPerRow = kChunkK / 8;                     // 16B vectors per smem row
    constexpr int kKSplitCTAs = Cfg::kKSplitCTAs;
    const int warp = threadIdx.x >> 5;
    const int m_tile = warp % kMTiles, k_slice = warp / kMTiles;
    const int expert_base = (blockIdx.x / kKSplitCTAs) * kExpertsPerCTA;
    const int k_part = blockIdx.x % kKSplitCTAs;
    const int m_pad = kMTiles * 16;
    const int chunks_per_part = h / kChunkK / kKSplitCTAs;
    const int chunk_base = k_part * chunks_per_part;
    const int num_chunks = chunks_per_part;
    float* part_logits = logits + static_cast<int64_t>(k_part) * m * e;
    const uint64_t w_policy = w_hint ? l2_evict_last_policy() : 0ull;

    // Padding token rows (>= m) are never loaded; zero them once in every stage.
    for (int st = 0; st < kStages; ++st)
        for (int i = threadIdx.x; i < (m_pad - m) * kVecPerRow; i += kThreads) {
            const int r = m + i / kVecPerRow, c = (i % kVecPerRow) * 8;
            *reinterpret_cast<uint4*>(&stage_h(st)[r][c]) = make_uint4(0, 0, 0, 0);
        }
    __syncthreads();

    auto issue_chunk = [&](int chunk) {
        const int st = chunk % kStages, k0 = (chunk_base + chunk) * kChunkK;
        for (int i = threadIdx.x; i < m * kVecPerRow; i += kThreads) {
            const int r = i / kVecPerRow, c = (i % kVecPerRow) * 8;
            cp_async_16(&stage_h(st)[r][c], hidden + static_cast<int64_t>(r) * h + k0 + c);
        }
        for (int i = threadIdx.x; i < kExpertsPerCTA * kVecPerRow; i += kThreads) {
            const int r = i / kVecPerRow, c = (i % kVecPerRow) * 8;
            const __nv_bfloat16* src = router_weight + static_cast<int64_t>(expert_base + r) * h + k0 + c;
            if (w_hint) cp_async_16_hint(&stage_w(st)[r][c], src, w_policy);
            else cp_async_16(&stage_w(st)[r][c], src);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int c = 0; c < kStages - 1; ++c) {
        if (c < num_chunks) issue_chunk(c);
        cp_async_commit();
    }
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        if (chunk + kStages - 1 < num_chunks) issue_chunk(chunk + kStages - 1);
        cp_async_commit();
        cp_async_wait<kStages - 1>();     // chunk `chunk` has landed for this thread
        __syncthreads();                  // ... and for every thread
        if (chunk == 0) stamp(stamps, 1);
        const int st = chunk % kStages;
        #pragma unroll
        for (int kk = 0; kk < kKPerWarp; kk += 16) {
            const int k = k_slice * kKPerWarp + kk;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(a, &stage_h(st)[m_tile * 16][k], kSmemLd);
            wmma::load_matrix_sync(b, &stage_w(st)[0][k], kSmemLd);   // B(k, n) = w_s[n][k]
            wmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();                  // stage `st` may be refilled next iteration
    }
    wmma::store_matrix_sync(part_s[warp], acc, 16, wmma::mem_row_major);
    __syncthreads();
    stamp(stamps, 2);
    // Reduce the kKSplit K-slices of each m-tile; 256 threads cover 16x16 per m-tile.
    for (int i = threadIdx.x; i < kMTiles * 256; i += kThreads) {
        const int tile = i / 256, r = (i % 256) / 16, c = i % 16;
        float v = 0.0f;
        #pragma unroll
        for (int ks = 0; ks < kKSplit; ++ks) v += part_s[ks * kMTiles + tile][r * 16 + c];
        const int t = tile * 16 + r, ex = expert_base + c;
        if (t < m && ex < e) part_logits[static_cast<int64_t>(t) * e + ex] = v;   // fp32 partial
    }
}

// Top-k (largest value, smallest expert id on ties) + softmax over the selected
// bf16 logits. One warp per token, lane owns experts lane, lane+32, ...
__device__ __forceinline__ void topk_softmax_token(
        const float* __restrict__ logits, int64_t* __restrict__ topk_idx,
        float* __restrict__ topk_weights, int t, int m, int e, int topk, int k_split) {
    const int lane = threadIdx.x & 31;
    constexpr int kPerLane = kMaxExperts / 32;
    {
        float v[kPerLane];
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int ex = lane + 32 * i;
            float acc = 0.0f;
            for (int ks = 0; ks < k_split; ++ks)
                acc += __ldcg(logits + (static_cast<int64_t>(ks) * m + t) * e + ex);
            v[i] = ex < e ? round_bf16(acc) : -INFINITY;
        }
        float sel_v[kMaxTopK];
        int sel_i[kMaxTopK];
        for (int k = 0; k < topk; ++k) {
            float best = -INFINITY; int best_i = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < kPerLane; ++i) {
                const int ex = lane + 32 * i;
                if (v[i] > best || (v[i] == best && ex < best_i)) { best = v[i]; best_i = ex; }
            }
            // warp argmax with smallest-id tie break
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
                if (ob > best || (ob == best && oi < best_i)) { best = ob; best_i = oi; }
            }
            sel_v[k] = best; sel_i[k] = best_i;
            if ((best_i & 31) == lane) v[best_i >> 5] = -INFINITY;
        }
        float mx = sel_v[0];
        for (int k = 1; k < topk; ++k) mx = fmaxf(mx, sel_v[k]);
        float sum = 0.0f, ex_v[kMaxTopK];
        for (int k = 0; k < topk; ++k) { ex_v[k] = expf(sel_v[k] - mx); sum += ex_v[k]; }
        if (lane < topk) {
            topk_idx[static_cast<int64_t>(t) * topk + lane] = sel_i[lane];
            topk_weights[static_cast<int64_t>(t) * topk + lane] = ex_v[lane] / sum;
        }
    }
}

// Tiny-M top-k (kTiny): same selection rule and same fp32 math as
// topk_softmax_token, restructured for latency:
//   * the kKSplit x 16 partial-logit loads are all issued before the first use
//     (compile-time K-split, no serial dependent loop) -> one L2 round trip;
//   * each candidate is a 32-bit key = orderable(bf16 logit) << 16 | (0xFFFF - id);
//     the logits are bf16-rounded so the key order is exactly (value desc, id asc)
//     and a round is one 32-bit warp max (5 shuffles) instead of paired
//     value/index shuffles + branches. -0.0 is canonicalised to +0.0 first
//     (float compare treats them equal; the legacy path tie-breaks them by id).
__device__ __forceinline__ uint32_t topk_key(float v, int ex) {
    v = v + 0.0f;                                                 // -0.0 -> +0.0
    const uint32_t b = __bfloat16_as_ushort(__float2bfloat16_rn(v));   // exact: v is bf16
    const uint32_t o = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
    return (o << 16) | (0xFFFFu - static_cast<uint32_t>(ex));
}
__device__ __forceinline__ float topk_key_value(uint32_t key) {
    const uint32_t o = key >> 16;
    const uint32_t b = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
    return __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(b)));
}
__device__ __forceinline__ int topk_key_index(uint32_t key) {
    return static_cast<int>(0xFFFFu - (key & 0xFFFFu));
}
__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Called by the whole CTA (kThreads): the kKSplit partial logits of every expert
// are fetched by all 256 threads (<= 2 experts per thread, one L2 round trip),
// summed in the legacy order and parked as keys in smem; warp 0 then selects.
template <int kKSplit, int kPerLane, int kTopK>
__device__ __forceinline__ void topk_softmax_token_tiny(
        const float* __restrict__ logits, int64_t* __restrict__ topk_idx,
        float* __restrict__ topk_weights, unsigned long long* stamps, int t, int m, int e) {
    __shared__ uint32_t key_s[kMaxExperts];
    const int lane = threadIdx.x & 31;
    for (int ex = threadIdx.x; ex < kPerLane * 32; ex += kThreads) {
        float acc = 0.0f;
        if (ex < e) {
            float p[kKSplit];
            #pragma unroll
            for (int ks = 0; ks < kKSplit; ++ks)
                p[ks] = __ldcg(logits + (static_cast<int64_t>(ks) * m + t) * e + ex);
            #pragma unroll
            for (int ks = 0; ks < kKSplit; ++ks) acc += p[ks];      // same order as legacy
        }
        key_s[ex] = topk_key(ex < e ? round_bf16(acc) : -INFINITY, ex);
    }
    __syncthreads();
    stamp(stamps, 4);
    if (threadIdx.x >= 32) return;
    uint32_t key[kPerLane];
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) key[i] = key_s[lane + 32 * i];
    // Everything below is fully unrolled (compile-time kTopK) so sel_*/ex_v stay in registers.
    float sel_v[kTopK];
    int sel_i[kTopK];
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        uint32_t best = 0;
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) best = max(best, key[i]);
        best = warp_max_u32(best);
        sel_v[k] = topk_key_value(best); sel_i[k] = topk_key_index(best);
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) key[i] = key[i] == best ? 0u : key[i];
    }
    stamp(stamps, 5);
    float mx = sel_v[0];
    #pragma unroll
    for (int k = 1; k < kTopK; ++k) mx = fmaxf(mx, sel_v[k]);
    float sum = 0.0f, ex_v[kTopK];
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) { ex_v[k] = expf(sel_v[k] - mx); sum += ex_v[k]; }
    int my_i = 0; float my_w = 0.0f;
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) if (lane == k) { my_i = sel_i[k]; my_w = ex_v[k] / sum; }
    if (lane < kTopK) {
        topk_idx[static_cast<int64_t>(t) * kTopK + lane] = my_i;
        topk_weights[static_cast<int64_t>(t) * kTopK + lane] = my_w;
    }
}

// ----------------------------------------------------------------- quant CTA
// mode 0: FP8 E4M3 per K128 group (16 lanes x 8 values), sf = amax / 448.
// mode 1: INT8 whole row, sf = amax / 127 replicated into every K128 slot.
template <int kMode>
__device__ __forceinline__ void quant_role(
        const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
        float* __restrict__ x_sf, int token, int h) {
    __shared__ float smem_warp_max[kThreads / 32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const __nv_bfloat16* xrow = hidden + static_cast<int64_t>(token) * h;
    uint8_t* qrow = x_bytes + static_cast<int64_t>(token) * h;
    float* sfrow = x_sf + static_cast<int64_t>(token) * (h / 128);
    float row_scale = 0.0f;
    if constexpr (kMode == 1) {
        float local = 0.0f;
        for (int k0 = threadIdx.x * 8; k0 < h; k0 += kThreads * 8) {
            float x[8]; unpack8(*reinterpret_cast<const uint4*>(xrow + k0), x);
            #pragma unroll
            for (int j = 0; j < 8; ++j) local = fmaxf(local, fabsf(x[j]));
        }
        local = warp_max(local);
        if (lane == 0) smem_warp_max[warp] = local;
        __syncthreads();
        float v = lane < kThreads / 32 ? smem_warp_max[lane] : 0.0f;
        v = warp_max(v);
        row_scale = fmaxf(v * (1.0f / 127.0f), 1.0e-30f);
        for (int g = threadIdx.x; g < h / 128; g += kThreads) sfrow[g] = row_scale;
    }
    // Every (warp, iteration) covers 256 K = two K128 groups; half-warp = one group.
    for (int k0 = threadIdx.x * 8; k0 < h; k0 += kThreads * 8) {
        float x[8]; unpack8(*reinterpret_cast<const uint4*>(xrow + k0), x);
        float scale;
        if constexpr (kMode == 0) {
            float local = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) local = fmaxf(local, fabsf(x[j]));
            local = half_warp_max(local);
            scale = fmaxf(local * (1.0f / 448.0f), 1.0e-30f);
            if ((lane & 15) == 0) sfrow[k0 / 128] = scale;
        } else {
            scale = row_scale;
        }
        const float inv = 1.0f / scale;
        uint2 packed;
        if constexpr (kMode == 0) {
            const uint16_t q0 = cvt_e4m3x2(x[0] * inv, x[1] * inv), q1 = cvt_e4m3x2(x[2] * inv, x[3] * inv);
            const uint16_t q2 = cvt_e4m3x2(x[4] * inv, x[5] * inv), q3 = cvt_e4m3x2(x[6] * inv, x[7] * inv);
            packed.x = static_cast<uint32_t>(q0) | (static_cast<uint32_t>(q1) << 16);
            packed.y = static_cast<uint32_t>(q2) | (static_cast<uint32_t>(q3) << 16);
        } else {
            uint32_t b[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int q = __float2int_rn(x[j] * inv);
                q = q > 127 ? 127 : (q < -127 ? -127 : q);
                b[j] = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
            }
            packed.x = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
            packed.y = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24);
        }
        *reinterpret_cast<uint2*>(qrow + k0) = packed;
    }
}

// ------------------------------------------------------------ full-K router
// DG_FE_TINYM_GRID=auto|N (m <= 16): the 96-CTA 4-way K-split above was sized
// for a 132-SM H100; on a 78-SM H20 97 CTAs leave 19 SMs double-occupied and
// the top-k CTA waits for the slowest of them. Here the router CTA count is
// derived from the SM count (T = SM count, or N): experts_per_cta (epc) =
// ceil(e / (T - 1)), router CTAs = ceil(e / epc) (H20: 5 experts x 77 CTAs +
// 1 merger CTA = 78 = one CTA per SM). Each CTA owns its experts over the FULL
// K, so it produces FINAL logits: no K-part partial reduction, no ticket.
//   * the whole K of every row (m hidden rows + epc weight rows) is issued up
//     front as TMA 1D bulk copies into row-contiguous smem (kFullKLd stride): 1-3
//     pieces per row (mbarrier per piece, <= ~24 copies per CTA -- the bulk-copy
//     queue is shallow), so only the first piece's HBM round trip is exposed and
//     WMMA on piece i overlaps the landing of pieces i+1..;
//   * accumulation order: warp w owns K-slice [32w, 32w+32) of every 256-chunk
//     and accumulates the 12 chunks (24 WMMA m16n16k16 bf16 -> fp32 steps, chunk
//     order) in ONE fp32 accumulator; the 8 warp partials are then summed in
//     fixed order ks = 0..7 through smem; the fp32 logit is rounded to bf16 once.
//     (legacy: 3 chunks -> 8-warp smem reduce -> 4 K-part partials summed ks =
//     0..3 in the top-k CTA -> bf16.) Deterministic, not bit-identical to legacy.
//   * hand-off: the CTA writes its <= 8 experts' logits as 32-bit top-k keys
//     (topk_key: value desc, id asc) into cand[token][cta][8]; unused slots get
//     the sentinel 1 (loses to every real key). A slot is 0 until written, so the
//     merger polls the slots themselves (ld.cv) -- the data is the flag, no
//     fence / release counter on the critical path.
//   * merger CTA (blockIdx == num_router_ctas, warp w -> tokens w, w+8): keeps a
//     running top-8 (lanes 0..7) and merges every batch of newly-arrived CTAs
//     (8 warp-max rounds over the lane-resident keys) in arrival order, so after
//     the last straggler lands only one small merge remains; softmax in the
//     legacy k = 0..7 order; then it zeroes the slots for the next launch.
//   * quantisation: router CTA t < m quantises token t between issuing its
//     chunk loads and consuming them (overlapped with the chunk-0 round trip).
// Row-contiguous smem: row r = one token (r < m) or one expert (m <= r < m + n_exp)
// over the full K, stride kFullKLd bf16 (6176 B: 32 B aligned for WMMA, 2-way bank
// spread), so a TMA piece of a row is one contiguous copy. 16 rows of tail after the
// weight rows keep the (discarded) B-tile rows inside the allocation.
constexpr int kFullKLd = 3072 + 16;
__host__ __device__ constexpr int fullk_smem_bytes(int m, int /*epc*/) {
    return (m + 16) * kFullKLd * 2 + (kThreads / 32) * 256 * 4;
}
__device__ __forceinline__ uint32_t smid_u32() {
    uint32_t v;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
    return v;
}
// stamp slot 6 of every CTA = %smid (placement / packing diagnostics, not a time)
__device__ __forceinline__ void stamp_smid(unsigned long long* stamps) {
    if (stamps != nullptr && threadIdx.x == 0) stamps[blockIdx.x * kStampSlots + 6] = smid_u32();
}
// TMA 1D bulk copies (cp.async.bulk, sm_90): one instruction per 512 B row-chunk,
// completion tracked by an mbarrier tx-count, so the in-flight bytes are not
// bounded by the LSU's per-thread cp.async tracking (with cp.async the 12-chunk
// issue itself took ~3 us on H20: the loads were throttled at issue).
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n"
                 " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 " @!p bra WAIT_%=;\n}" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_bulk_g2s_hint(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t policy) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}
// One redux.sync instruction instead of a 5-level shuffle tree (~5x lower latency
// per round; the merger's 8 rounds are on the exposed tail of the kernel).
__device__ __forceinline__ uint32_t warp_max_u32_redux(uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_max_sync(0xffffffffu, v);
#else
    return warp_max_u32(v);
#endif
}
__device__ __forceinline__ uint32_t ld_cv_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

template <int kMode>
__device__ __forceinline__ void router_role_fullk(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint32_t* __restrict__ cand,           // [m][num_router_ctas][kCandSlots] keys
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        unsigned long long* stamps,
        int m, int h, int e, int epc, int num_router_ctas, int w_hint) {
    using namespace nvcuda;
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    auto row_s = [&](int r) { return reinterpret_cast<__nv_bfloat16*>(dyn_smem) + static_cast<int>(r) * kFullKLd; };
    float (*part_s)[256] = reinterpret_cast<float (*)[256]>(dyn_smem + (m + 16) * kFullKLd * 2);
    constexpr int kKSplit = kThreads / 32;                      // 8 warps = 8 K-slices per chunk
    constexpr int kKPerWarp = kChunkK / kKSplit;                // 32
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int expert_base = blockIdx.x * epc;
    const int n_exp = min(epc, e - expert_base);
    const int num_chunks = h / kChunkK;
    const int n_rows = m + n_exp;
    // TMA pieces per row: the bulk-copy queue is shallow (72 x 512 B copies per CTA
    // took ~2.5 us to *issue* on H20), so keep the copy count <= ~24: 3 pieces when
    // <= 6 rows, 2 when <= 12, else 1 (m = 16 -> 23 whole-row copies).
    int parts = n_rows <= 6 ? 3 : (n_rows <= 12 ? 2 : 1);
    while (num_chunks % parts) --parts;
    const int chunks_per_part = num_chunks / parts;
    const uint32_t piece_bytes = static_cast<uint32_t>(chunks_per_part * kChunkK * 2);
    const uint64_t w_policy = w_hint ? l2_evict_last_policy() : 0ull;
    // Rows >= m of the A tile and >= n_exp of the B tile are never loaded: they
    // read whatever the neighbouring rows hold, which only feeds output rows /
    // columns that are discarded (C[r][c] depends on A row r and B column c only).
    __shared__ __align__(8) uint64_t piece_bar[3];
    if (threadIdx.x < parts) mbar_init(&piece_bar[threadIdx.x], 1);
    mbar_fence_init();
    __syncthreads();
    stamp(stamps, 7);        // prologue done (mbarrier init + fence + sync), about to issue
    if (warp == 0) {
        if (lane < parts) mbar_arrive_expect_tx(&piece_bar[lane], static_cast<uint32_t>(n_rows) * piece_bytes);
        __syncwarp();
        // flat (piece, row) list spread over the 32 lanes: <= 3 copies per lane
        for (int i = lane; i < parts * n_rows; i += 32) {
            const int pc = i / n_rows, r = i % n_rows, k0 = pc * chunks_per_part * kChunkK;
            const __nv_bfloat16* src = r < m ? hidden + static_cast<int64_t>(r) * h + k0
                                             : router_weight + static_cast<int64_t>(expert_base + r - m) * h + k0;
            if (w_hint && r >= m) tma_bulk_g2s_hint(row_s(r) + k0, src, piece_bytes, &piece_bar[pc], w_policy);
            else tma_bulk_g2s(row_s(r) + k0, src, piece_bytes, &piece_bar[pc]);
        }
    }
    stamp(stamps, 5);
    // Overlapped activation quantisation: while this CTA's pieces are in flight.
    if (static_cast<int>(blockIdx.x) < m) {
        quant_role<kMode>(hidden, x_bytes, x_sf, blockIdx.x, h);
        stamp(stamps, 4);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        if (chunk % chunks_per_part == 0) {
            mbar_wait_parity(&piece_bar[chunk / chunks_per_part], 0u);   // piece landed (async proxy -> all threads)
            if (chunk == 0) stamp(stamps, 1);
        }
        #pragma unroll
        for (int kk = 0; kk < kKPerWarp; kk += 16) {
            const int k = chunk * kChunkK + warp * kKPerWarp + kk;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(a, row_s(0) + k, kFullKLd);
            wmma::load_matrix_sync(b, row_s(m) + k, kFullKLd);   // B(k, n) = w_s[n][k]
            wmma::mma_sync(acc, a, b, acc);
        }
    }
    wmma::store_matrix_sync(part_s[warp], acc, 16, wmma::mem_row_major);
    __syncthreads();
    stamp(stamps, 2);
    // 256 threads = 16 token rows x 16 expert columns: fixed-order 8-slice sum,
    // bf16 round, 32-bit key, straight into this CTA's candidate slots.
    {
        const int r = threadIdx.x / 16, c = threadIdx.x % 16;
        if (r < m && c < kCandSlots) {
            float v = 0.0f;
            #pragma unroll
            for (int ks = 0; ks < kKSplit; ++ks) v += part_s[ks][r * 16 + c];
            const int ex = expert_base + c;
            const uint32_t key = c < n_exp ? topk_key(round_bf16(v), ex) : 1u;
            cand[(static_cast<int64_t>(r) * num_router_ctas + blockIdx.x) * kCandSlots + c] = key;
        }
    }
}

// Merger CTA: one warp per token (m <= 16 -> <= 2 tokens per warp). Lane l owns
// slots l, l+32, ... of cand[token] (coalesced 128 B polls; a CTA's 8 keys land in
// 8 consecutive lanes). Register-lean streaming merge (no per-slot key array -> no
// local-memory spills): every poll round folds the newly-arrived keys into a
// per-lane sorted top-8 `loc[]` (insertion), then, unless the batch maximum cannot
// enter the running top-8 (early exit, the common case late in the stream), runs
// 8 rounds of one redux.sync max over (running key in lanes 0..7, loc[0]) with the
// winner popped -> new running top-8 in lanes 0..7, descending.
template <int kTopK, int kSlotsPerLane>
__device__ __forceinline__ void merger_role(
        uint32_t* __restrict__ cand, int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        unsigned long long* stamps, int m, int num_router_ctas) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nslots = num_router_ctas * kCandSlots;
    for (int t = warp; t < m; t += kThreads / 32) {
        uint32_t* base = cand + static_cast<int64_t>(t) * nslots;
        uint32_t pending = 0u;
        #pragma unroll
        for (int i = 0; i < kSlotsPerLane; ++i)
            if (lane + 32 * i < nslots) pending |= 1u << i;
        uint32_t run = 0u;                 // lanes 0..7: running top-8 (descending), else 0
        uint32_t loc[kTopK];               // this lane's sorted (desc) keys of the current batch
        #pragma unroll
        for (int j = 0; j < kTopK; ++j) loc[j] = 0u;
        bool first = true;
        // Software-pipelined polling: the loads of round r+1 are issued before round r's
        // keys are folded/merged, so the merge work hides under the L2 round trip and a
        // key written during the merge is picked up by the already in-flight round.
        uint32_t v[kSlotsPerLane];
        #pragma unroll
        for (int i = 0; i < kSlotsPerLane; ++i)
            v[i] = (pending & (1u << i)) ? ld_cv_u32(base + lane + 32 * i) : 0u;
        while (true) {
            uint32_t pend_next = pending;
            #pragma unroll
            for (int i = 0; i < kSlotsPerLane; ++i)
                if (v[i] != 0u) pend_next &= ~(1u << i);
            uint32_t vn[kSlotsPerLane];
            #pragma unroll
            for (int i = 0; i < kSlotsPerLane; ++i)
                vn[i] = (pend_next & (1u << i)) ? ld_cv_u32(base + lane + 32 * i) : 0u;
            bool got = false;
            #pragma unroll
            for (int i = 0; i < kSlotsPerLane; ++i) {
                if (v[i] != 0u) {
                    pending &= ~(1u << i); got = true;
                    uint32_t k = v[i];
                    #pragma unroll
                    for (int j = 0; j < kTopK; ++j) {       // sorted insertion, drops the smallest
                        const uint32_t lo = min(k, loc[j]);
                        loc[j] = max(k, loc[j]); k = lo;
                    }
                }
            }
            if (__any_sync(0xffffffffu, got)) {
                if (t == 0) { if (first) stamp(stamps, 1); stamp(stamps, 2); }
                first = false;
                const uint32_t batch_max = warp_max_u32_redux(loc[0]);
                const uint32_t run8 = __shfl_sync(0xffffffffu, run, kTopK - 1);   // current 8th best (0 if < 8 yet)
                if (batch_max > run8) {
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
                #pragma unroll
                for (int j = 0; j < kTopK; ++j) loc[j] = 0u;   // losers can never re-enter
            }
            if (__all_sync(0xffffffffu, pending == 0u)) break;
            #pragma unroll
            for (int i = 0; i < kSlotsPerLane; ++i) v[i] = vn[i];
        }
        if (t == 0) stamp(stamps, 3);
        // softmax over the 8 selected bf16 logits, legacy order (k = 0..7 sequential sum)
        const float sel_v = lane < kTopK ? topk_key_value(run) : -INFINITY;
        const float mx = warp_max(sel_v);
        const float ex = lane < kTopK ? expf(sel_v - mx) : 0.0f;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) sum += __shfl_sync(0xffffffffu, ex, k);
        if (lane < kTopK) {
            topk_idx[static_cast<int64_t>(t) * kTopK + lane] = topk_key_index(run);
            topk_weights[static_cast<int64_t>(t) * kTopK + lane] = ex / sum;
        }
        if (t == 0) stamp(stamps, 4);
        // reset the slots for the next launch (every writer has been consumed)
        #pragma unroll
        for (int i = 0; i < kSlotsPerLane; ++i)
            if (lane + 32 * i < nslots) base[lane + 32 * i] = 0u;
    }
}

// kTiny: <= 85 regs/thread so 3 CTAs (59 KB smem each) fit per SM -> single wave.
// kFullK: <= 128 regs (merger warp holds 32 keys), 2 CTAs/SM cap; grid <= SM count anyway.
template <int kMTiles, int kMode, bool kTiny, bool kFullK>
__global__ void __launch_bounds__(kThreads, kFullK ? 2 : (kTiny ? 3 : 1)) router_quant_topk_kernel(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        int* __restrict__ ticket, float* __restrict__ logits,
        unsigned long long* stamps,
        int m, int h, int e, int topk, int num_router_ctas, int w_hint, int pdl_mode, int epc) {
    stamp(stamps, 0);
    stamp_smid(stamps);
    if constexpr (kFullK) {
        if (pdl_mode == 1) pdl_trigger();
        uint32_t* cand = reinterpret_cast<uint32_t*>(logits);
        if (static_cast<int>(blockIdx.x) >= num_router_ctas) {
            if (num_router_ctas * kCandSlots <= 20 * 32)      // H20: 77 x 8 = 616 slots -> 20 per lane
                merger_role<kMaxTopK, 20>(cand, topk_idx, topk_weights, stamps, m, num_router_ctas);
            else
                merger_role<kMaxTopK, kMaxSlotsPerLane>(cand, topk_idx, topk_weights, stamps, m, num_router_ctas);
        } else {
            router_role_fullk<kMode>(hidden, router_weight, cand, x_bytes, x_sf, stamps, m, h, e, epc, num_router_ctas, w_hint);
            stamp(stamps, 3);
        }
        if (pdl_mode == 2) pdl_trigger();
        return;
    }
    // PDL (DG_FE_PDL): 1 = trigger at CTA start (the dependent MegaMoE grid is
    // scheduled as soon as every FE CTA is resident), 2 = trigger after this CTA's
    // last store (only the launch latency overlaps). The dependent's
    // griddepcontrol.wait still blocks until this grid has fully completed and
    // flushed, so nothing it reads can be early. No-op without a PDL dependent.
    if (pdl_mode == 1) pdl_trigger();
    if (static_cast<int>(blockIdx.x) >= num_router_ctas) {
        // Quant CTA: quantize this token, then wait for all router CTAs (they must
        // be co-resident: kTiny keeps 96 + m CTAs within 78 SMs x 3 CTAs) and run
        // top-k for this token.
        const int token = blockIdx.x - num_router_ctas;
        quant_role<kMode>(hidden, x_bytes, x_sf, token, h);
        stamp(stamps, 1);
        if constexpr (kTiny) {
            if (threadIdx.x == 0) {
                while (*reinterpret_cast<volatile int*>(ticket) < num_router_ctas) { }
                __threadfence();
            }
            __syncthreads();
            stamp(stamps, 2);
            constexpr int kKS = RouterCfg<kMTiles, kTiny>::kKSplitCTAs;
            if (e <= 384)   // E=384 -> 12 experts per lane
                topk_softmax_token_tiny<kKS, 12, kMaxTopK>(logits, topk_idx, topk_weights, stamps, token, m, e);
            else
                topk_softmax_token_tiny<kKS, kMaxExperts / 32, kMaxTopK>(logits, topk_idx, topk_weights, stamps, token, m, e);
            // warps 1..7 returned inside topk_softmax_token_tiny; warp 0 continues
            stamp(stamps, 3);
            if (pdl_mode == 2) pdl_trigger();
            if (threadIdx.x == 0 && atomicAdd(ticket + 1, 1) == m - 1) {
                ticket[1] = 0;
                __threadfence();
                ticket[0] = 0;
            }
            return;
        }
        if (threadIdx.x < 32) {
            if (threadIdx.x == 0) {
                while (*reinterpret_cast<volatile int*>(ticket) < num_router_ctas) __nanosleep(200);
                __threadfence();
            }
            __syncwarp();
            stamp(stamps, 2);
            topk_softmax_token(logits, topk_idx, topk_weights, token, m, e, topk, RouterCfg<kMTiles, kTiny>::kKSplitCTAs);
            stamp(stamps, 3);
            if (pdl_mode == 2) pdl_trigger();
            // Last token CTA resets the counters for the next launch.
            if (threadIdx.x == 0 && atomicAdd(ticket + 1, 1) == m - 1) {
                ticket[1] = 0;
                __threadfence();
                ticket[0] = 0;
            }
        }
        return;
    }
    router_role<kMTiles, kTiny>(hidden, router_weight, logits, stamps, m, h, e, w_hint);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(ticket, 1);
    stamp(stamps, 3);
    if (pdl_mode == 2) pdl_trigger();
}

// DG_FE_ROUTER_L2_PERSIST=1: CUDA persisting-L2 set-aside for the router weights.
// Once per process: carve `bytes` (clamped to cudaDevAttrMaxPersistingL2CacheSize)
// out of L2; per launch: an access-policy-window launch attribute over the router
// weight buffer (hitProp Persisting). Launch attributes are recorded on the kernel
// node under stream capture, so eager and CUDA-graph paths behave the same. Normal
// / streaming accesses (MoE weights, L2 flushes) cannot evict persisting lines.
static void set_persisting_l2_once(size_t bytes) {
    static bool done = false;
    if (done) return;
    done = true;
    int dev = 0, max_persist = 0, max_window = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev);
    cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev);
    const size_t want = std::min(bytes, static_cast<size_t>(max_persist));
    const cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want);
    size_t got = 0;
    cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize);
    if (getenv("DG_FE_ROUTER_L2_PERSIST_VERBOSE") != nullptr)
        fprintf(stderr, "[fable_frontend] persisting L2: max %d B, max window %d B, requested %zu B, "
                "set %zu B (%s)\n", max_persist, max_window, want, got, cudaGetErrorString(err));
}

static int num_sms_cached() {
    static int n = 0;
    if (n == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        if (n <= 0) n = 1;
    }
    return n;
}
// Full-K plan: target T CTAs in total (T = SM count when grid <= 0, else grid);
// one merger CTA, epc = ceil(e / (T - 1)) clamped to [1, kCandSlots], router
// CTAs = ceil(e / epc) (the last one may own fewer experts).
static void fullk_plan(int e, int grid, int& epc, int& router_ctas) {
    const int target = (grid <= 0 ? num_sms_cached() : grid) - 1;
    epc = (e + std::max(target, 1) - 1) / std::max(target, 1);
    epc = std::min(std::max(epc, 1), kCandSlots);
    router_ctas = (e + epc - 1) / epc;
}

template <int kMTiles, int kMode, bool kTiny, bool kFullK>
void launch(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
            int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
            int m, int h, int e, int topk, int l2_persist, int pdl_mode, int grid, cudaStream_t stream) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    int epc = kExpertsPerCTA, router_ctas = ((e + kExpertsPerCTA - 1) / kExpertsPerCTA) * Cfg::kKSplitCTAs;
    int smem_bytes = Cfg::kDynSmemBytes, num_ctas = router_ctas + m;
    if constexpr (kFullK) {
        fullk_plan(e, grid, epc, router_ctas);
        smem_bytes = fullk_smem_bytes(m, epc);
        num_ctas = router_ctas + 1;
        // DG_FE_FULLK_1PERSM (default 1): when the grid fits the SM count, request
        // enough dynamic smem that only one CTA fits per SM, so the block scheduler
        // cannot pack two router CTAs (and their in-flight loads) onto one SM.
        static const int one_per_sm = getenv("DG_FE_FULLK_1PERSM") ? atoi(getenv("DG_FE_FULLK_1PERSM")) : 1;
        if (one_per_sm && num_ctas <= num_sms_cached()) smem_bytes = std::max(smem_bytes, 116 * 1024);
    }
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(router_quant_topk_kernel<kMTiles, kMode, kTiny, kFullK>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kFullK ? std::max(fullk_smem_bytes(16, kCandSlots), 116 * 1024) : Cfg::kDynSmemBytes);
        attr_set = true;
    }
    const size_t w_bytes = static_cast<size_t>(e) * h * sizeof(__nv_bfloat16);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(num_ctas);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    cfg.attrs = attrs;
    cfg.numAttrs = 0;
    if (l2_persist == 1) {
        set_persisting_l2_once(w_bytes);
        int max_window = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev);
        auto& a = attrs[cfg.numAttrs++];
        a.id = cudaLaunchAttributeAccessPolicyWindow;
        a.val.accessPolicyWindow.base_ptr = const_cast<void*>(static_cast<const void*>(w));
        a.val.accessPolicyWindow.num_bytes = std::min(w_bytes, static_cast<size_t>(max_window));
        a.val.accessPolicyWindow.hitRatio = 1.0f;
        a.val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        a.val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    }
    const int w_hint = l2_persist == 2 ? 1 : 0;
    cudaLaunchKernelEx(&cfg, router_quant_topk_kernel<kMTiles, kMode, kTiny, kFullK>,
                       hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, router_ctas, w_hint, pdl_mode, epc);
}

template <int kMode>
void launch_mode(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
                 int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
                 int m, int h, int e, int topk, bool tiny, int l2_persist, int pdl_mode, int grid, cudaStream_t stream) {
    if (m <= 16 && tiny && topk == kMaxTopK && grid != 96 && h % kChunkK == 0 && h / kChunkK <= kFullKStages)
        launch<1, kMode, true, true>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, stream);
    else if (m <= 16 && tiny && topk == kMaxTopK) launch<1, kMode, true, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, stream);
    else if (m <= 16) launch<1, kMode, false, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, stream);
    else if (m <= 32) launch<2, kMode, false, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, stream);
    else launch<4, kMode, false, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, stream);
}

}  // namespace

size_t router_quant_topk_frontend_workspace_bytes(int e) {
    return kFrontendStampsOffsetBase + static_cast<size_t>(4) * 64 * e * 4 + kFrontendStampsBytes;
}

int router_quant_topk_frontend_router_ctas(int m, int h, int e, int topk, int tiny, int grid) {
    if (m <= 16 && tiny && topk == kMaxTopK && grid != 96 && h % kChunkK == 0 && h / kChunkK <= kFullKStages) {
        int epc = 0, router_ctas = 0;
        fullk_plan(e, grid, epc, router_ctas);
        return router_ctas;
    }
    const int groups = (e + kExpertsPerCTA - 1) / kExpertsPerCTA;
    return groups * (m <= 16 ? 4 : 2);
}

void launch_router_quant_topk_frontend(
        const void* hidden, const void* router_weight,
        void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
        void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
        int tiny, int stamps_on, int l2_persist, int pdl_mode, int grid, cudaStream_t stream) {
    int* ticket = static_cast<int*>(workspace);
    float* logits = reinterpret_cast<float*>(static_cast<char*>(workspace) + kFrontendStampsOffsetBase);
    unsigned long long* stamps = nullptr;
    if (stamps_on && workspace_bytes >= router_quant_topk_frontend_workspace_bytes(e))
        stamps = reinterpret_cast<unsigned long long*>(
            static_cast<char*>(workspace) + kFrontendStampsOffsetBase + static_cast<size_t>(4) * 64 * e * 4);
    const auto* hp = static_cast<const __nv_bfloat16*>(hidden);
    const auto* wp = static_cast<const __nv_bfloat16*>(router_weight);
    const bool use_tiny = tiny != 0;
    if (mode == 0)
        launch_mode<0>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, stamps, m, h, e, topk, use_tiny, l2_persist, pdl_mode, grid, stream);
    else
        launch_mode<1>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, stamps, m, h, e, topk, use_tiny, l2_persist, pdl_mode, grid, stream);
}
