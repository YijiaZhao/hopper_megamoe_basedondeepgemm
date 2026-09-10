// Fused router + top-k + activation-quantization frontend for the SM90 fused
// MegaMoE (see mega_frontend.h). One launch, two CTA roles:
//   * router CTAs (E / 8 of them): warp w owns expert 8*blockIdx + w, streams
//     its weight row once from HBM and accumulates the dot product with every
//     token (hidden rows are L2-resident). Logits are rounded to bf16 (matching
//     a bf16 router matmul) and parked in the workspace; the last router CTA to
//     finish (atomic ticket) runs top-k + softmax for all tokens.
//   * quant CTAs (m of them): one token each, per-K128 FP8 (mode 0) or whole
//     row INT8 (mode 1), scale computed online.
#include "fable_frontend.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>

namespace {

constexpr int kExpertsPerCTA = 16;
constexpr int kThreads = 256;
constexpr int kMaxTopK = 8;
constexpr int kMaxExperts = 512;

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
constexpr int kStampSlots = 8;
__device__ __forceinline__ void stamp(unsigned long long* stamps, int slot) {
    if (stamps != nullptr && threadIdx.x == 0) stamps[blockIdx.x * kStampSlots + slot] = globaltimer_ns();
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

template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_role(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        float* __restrict__ logits,          // [m, e] workspace
        unsigned long long* stamps,
        int m, int h, int e) {
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
            cp_async_16(&stage_w(st)[r][c],
                        router_weight + static_cast<int64_t>(expert_base + r) * h + k0 + c);
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
    // CTA-wide radix select of the kTopK-th largest key (4 passes x 8 bits, 256-bin
    // smem histogram; keys are unique so the final bin holds exactly one key). Then
    // the <= kTopK keys >= threshold are gathered and ranked by warp 0. Same result as
    // kTopK rounds of (value desc, id asc) argmax; ~half the latency (no 40-deep
    // shuffle chain).
    __shared__ int hist[256];
    __shared__ int s_bin, s_need, s_cnt;
    __shared__ uint32_t s_sel[kTopK];
    const uint32_t k0 = key_s[threadIdx.x];
    const uint32_t k1 = threadIdx.x + kThreads < kPerLane * 32 ? key_s[threadIdx.x + kThreads] : 0u;
    static_assert(kPerLane * 32 <= 2 * kThreads, "two keys per thread");
    uint32_t prefix = 0, mask = 0;
    int need = kTopK;
    if (threadIdx.x == 0) s_cnt = 0;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        hist[threadIdx.x] = 0;
        __syncthreads();
        if ((k0 & mask) == prefix) atomicAdd(&hist[(k0 >> shift) & 255], 1);
        if (k1 != 0u && (k1 & mask) == prefix) atomicAdd(&hist[(k1 >> shift) & 255], 1);
        __syncthreads();
        if (threadIdx.x < 32) {
            int local[8], lsum = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) { local[j] = hist[8 * lane + j]; lsum += local[j]; }
            int run = lsum;                          // inclusive suffix sum over lanes >= lane
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_down_sync(0xffffffffu, run, o);
                if (lane + o < 32) run += v;
            }
            const int above = run - lsum;            // keys in bins above this lane's 8 bins
            if (above < need && above + lsum >= need) {
                int cum = above;
                #pragma unroll
                for (int j = 7; j >= 0; --j) {
                    if (cum + local[j] >= need) { s_bin = 8 * lane + j; s_need = need - cum; break; }
                    cum += local[j];
                }
            }
        }
        __syncthreads();
        prefix |= static_cast<uint32_t>(s_bin) << shift;
        mask |= 255u << shift;
        need = s_need;
        __syncthreads();
    }
    const uint32_t thresh = prefix;                  // exact key of the kTopK-th largest
    if (k0 >= thresh) s_sel[atomicAdd(&s_cnt, 1)] = k0;
    if (k1 != 0u && k1 >= thresh) s_sel[atomicAdd(&s_cnt, 1)] = k1;
    __syncthreads();
    stamp(stamps, 5);
    if (threadIdx.x >= 32) return;
    uint32_t cand[kTopK];
    #pragma unroll
    for (int j = 0; j < kTopK; ++j) cand[j] = s_sel[j];
    float sel_v[kTopK];
    int sel_i[kTopK];
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) {                // sel[k] = candidate of rank k (descending)
        uint32_t pick = 0;
        #pragma unroll
        for (int j = 0; j < kTopK; ++j) {
            int rank = 0;
            #pragma unroll
            for (int i = 0; i < kTopK; ++i) rank += cand[i] > cand[j];
            pick = rank == k ? cand[j] : pick;
        }
        sel_v[k] = topk_key_value(pick); sel_i[k] = topk_key_index(pick);
    }
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

// kTiny: <= 85 regs/thread so 3 CTAs (59 KB smem each) fit per SM -> single wave.
template <int kMTiles, int kMode, bool kTiny>
__global__ void __launch_bounds__(kThreads, kTiny ? 3 : 1) router_quant_topk_kernel(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        int* __restrict__ ticket, float* __restrict__ logits,
        unsigned long long* stamps,
        int m, int h, int e, int topk, int num_router_ctas) {
    stamp(stamps, 0);
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
            // Last token CTA resets the counters for the next launch.
            if (threadIdx.x == 0 && atomicAdd(ticket + 1, 1) == m - 1) {
                ticket[1] = 0;
                __threadfence();
                ticket[0] = 0;
            }
        }
        return;
    }
    router_role<kMTiles, kTiny>(hidden, router_weight, logits, stamps, m, h, e);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(ticket, 1);
    stamp(stamps, 3);
}

template <int kMTiles, int kMode, bool kTiny>
void launch(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
            int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
            int m, int h, int e, int topk, cudaStream_t stream) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    const int router_ctas = ((e + kExpertsPerCTA - 1) / kExpertsPerCTA) * Cfg::kKSplitCTAs;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(router_quant_topk_kernel<kMTiles, kMode, kTiny>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kDynSmemBytes);
        attr_set = true;
    }
    router_quant_topk_kernel<kMTiles, kMode, kTiny><<<router_ctas + m, kThreads, Cfg::kDynSmemBytes, stream>>>(
        hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, router_ctas);
}

template <int kMode>
void launch_mode(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
                 int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
                 int m, int h, int e, int topk, bool tiny, cudaStream_t stream) {
    if (m <= 16 && tiny && topk == kMaxTopK) launch<1, kMode, true>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, stream);
    else if (m <= 16) launch<1, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, stream);
    else if (m <= 32) launch<2, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, stream);
    else launch<4, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, stream);
}

}  // namespace

size_t router_quant_topk_frontend_workspace_bytes(int e) {
    return kFrontendStampsOffsetBase + static_cast<size_t>(4) * 64 * e * 4 + kFrontendStampsBytes;
}

void launch_router_quant_topk_frontend(
        const void* hidden, const void* router_weight,
        void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
        void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
        int tiny, int stamps_on, cudaStream_t stream) {
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
                       ticket, logits, stamps, m, h, e, topk, use_tiny, stream);
    else
        launch_mode<1>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, stamps, m, h, e, topk, use_tiny, stream);
}
