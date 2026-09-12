#pragma once
// Fable frontend device code (legacy router WMMA unit, tiny-M top-k + softmax) used by
// the standalone frontend kernel (csrc/fable_frontend.cu). Kernel-agnostic: callers pass
// the thread index inside the 256-thread CTA (`tid`), a CTA barrier (`sync`) and the smem
// region.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <math.h>

namespace deep_gemm::fable_fe {

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
// Optional per-CTA phase stamps (DG_FE_STAMPS=1): [blockIdx][8] u64 ns. Legacy grid:
// router CTA: 0 start / 1 chunk0 landed / 2 mma done / 3 ticket bumped / 6 %smid
// quant  CTA: 0 start / 1 quant done / 2 ticket seen / 3 top-k done
//             (tiny top-k: 4 partial logits loaded / 5 selection rounds done)
// (cc router slots: see router_cc_lean_kernel in csrc/fable_frontend.cu.)
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

// ---------------------------------------------------------------- router unit
// Tensor-core router GEMM: a unit owns kExpertsPerCTA (= 16) experts and all m
// tokens over one K-part. Hidden [m x 256] and weight [16 x 256] K-chunks are staged
// in smem; warp w computes the (m-tile, K-slice) product with WMMA bf16 m16n16k16 in
// fp32; K-slice partials are reduced through smem at the end. Hidden rows are
// read once per unit (L2-resident), weights once overall.
constexpr int kChunkK = 256;
constexpr int kSmemLd = kChunkK + 8;     // bf16 elements per smem row (bank-spread)
constexpr int kMaxMTiles = 4;            // m <= 64

// Pipeline depth per m-tile count: fewer token rows -> smaller stages -> deeper
// pipeline (the loop is HBM/L2 latency bound, compute is negligible).
// kTiny (m <= 16 only): 3 stages instead of 8. With 8 stages the
// dynamic smem is 143 KB -> 1 CTA/SM, so the 96 router CTAs + m quant CTAs do
// not fit on the 78 SMs of an H20 and run as two waves (the quant/top-k CTAs
// are in the second wave). 3 stages = 59 KB -> 3 CTAs/SM, single wave. H=3072
// needs exactly 3 chunks per K-part, so nothing is pipelined away. The math
// (WMMA order, K-split partial order, bf16 rounding) is unchanged -> bit-identical.
template <int kMTiles, bool kTiny = false> struct RouterCfg {
    static_assert(!kTiny || kMTiles == 1, "tiny-M config is the m <= 16 path");
    // K-split across units: partial logits go to workspace slice [ks][m][e] and are
    // summed in fixed order by the top-k (deterministic).
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
// Per-unit smem accessors (`dyn_smem` = RouterCfg::kDynSmemBytes bytes, 128 B aligned).
template <int kMTiles, bool kTiny>
struct RouterSmem {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    uint8_t* base;
    __device__ __forceinline__ __nv_bfloat16 (*h(int st))[kSmemLd] {
        return reinterpret_cast<__nv_bfloat16 (*)[kSmemLd]>(base + st * Cfg::kStageElems * 2);
    }
    __device__ __forceinline__ __nv_bfloat16 (*w(int st))[kSmemLd] {
        return reinterpret_cast<__nv_bfloat16 (*)[kSmemLd]>(base + (st * Cfg::kStageElems + Cfg::kHRows * kSmemLd) * 2);
    }
    __device__ __forceinline__ float (*part())[256] {
        return reinterpret_cast<float (*)[256]>(base + Cfg::kStages * Cfg::kStageElems * 2);
    }
};

// Zero the padding token rows (>= m) of every stage and issue chunk `chunk` of
// unit `unit_idx` into stage `chunk % kStages` (no commit).
template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_zero_padding(RouterSmem<kMTiles, kTiny> smem, int m, int tid) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    constexpr int kVecPerRow = kChunkK / 8;
    const int m_pad = kMTiles * 16;
    for (int st = 0; st < Cfg::kStages; ++st)
        for (int i = tid; i < (m_pad - m) * kVecPerRow; i += kThreads) {
            const int r = m + i / kVecPerRow, c = (i % kVecPerRow) * 8;
            *reinterpret_cast<uint4*>(&smem.h(st)[r][c]) = make_uint4(0, 0, 0, 0);
        }
}
template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_issue_chunk(
        RouterSmem<kMTiles, kTiny> smem,
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        int m, int h, int unit_idx, int chunk, int tid) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    constexpr int kVecPerRow = kChunkK / 8;
    const int expert_base = (unit_idx / Cfg::kKSplitCTAs) * kExpertsPerCTA;
    const int k_part = unit_idx % Cfg::kKSplitCTAs;
    const int chunks_per_part = h / kChunkK / Cfg::kKSplitCTAs;
    const int st = chunk % Cfg::kStages, k0 = (k_part * chunks_per_part + chunk) * kChunkK;
    for (int i = tid; i < m * kVecPerRow; i += kThreads) {
        const int r = i / kVecPerRow, c = (i % kVecPerRow) * 8;
        cp_async_16(&smem.h(st)[r][c], hidden + static_cast<int64_t>(r) * h + k0 + c);
    }
    for (int i = tid; i < kExpertsPerCTA * kVecPerRow; i += kThreads) {
        const int r = i / kVecPerRow, c = (i % kVecPerRow) * 8;
        cp_async_16(&smem.w(st)[r][c], router_weight + static_cast<int64_t>(expert_base + r) * h + k0 + c);
    }
}

// WMMA of one landed chunk (stage `st`) into `acc`; warp `warp` owns K-slice
// [warp / kMTiles * kKPerWarp, +kKPerWarp) of the chunk and m-tile warp % kMTiles.
template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_mma_chunk(
        RouterSmem<kMTiles, kTiny> smem, int st, int warp,
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& acc) {
    using namespace nvcuda;
    constexpr int kKSplit = (kThreads / 32) / kMTiles;          // warps per m-tile
    constexpr int kKPerWarp = kChunkK / kKSplit;                // 256 / {8,4,2}
    const int m_tile = warp % kMTiles, k_slice = warp / kMTiles;
    #pragma unroll
    for (int kk = 0; kk < kKPerWarp; kk += 16) {
        const int k = k_slice * kKPerWarp + kk;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, &smem.h(st)[m_tile * 16][k], kSmemLd);
        wmma::load_matrix_sync(b, &smem.w(st)[0][k], kSmemLd);   // B(k, n) = w_s[n][k]
        wmma::mma_sync(acc, a, b, acc);
    }
}

// Reduce the kKSplit K-slices of each m-tile (fixed warp order) and store the fp32
// partial logits of this unit's K-part: logits[(k_part * m + t) * e + ex].
template <int kMTiles, bool kTiny, typename Sync>
__device__ __forceinline__ void router_reduce_store(
        RouterSmem<kMTiles, kTiny> smem, float* __restrict__ logits,
        int m, int e, int unit_idx, int tid,
        const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& acc,
        const Sync& sync) {
    using namespace nvcuda;
    using Cfg = RouterCfg<kMTiles, kTiny>;
    constexpr int kKSplit = (kThreads / 32) / kMTiles;
    const int warp = tid >> 5;
    const int expert_base = (unit_idx / Cfg::kKSplitCTAs) * kExpertsPerCTA;
    const int k_part = unit_idx % Cfg::kKSplitCTAs;
    float* part_logits = logits + static_cast<int64_t>(k_part) * m * e;
    wmma::store_matrix_sync(smem.part()[warp], acc, 16, wmma::mem_row_major);
    sync();
    for (int i = tid; i < kMTiles * 256; i += kThreads) {
        const int tile = i / 256, r = (i % 256) / 16, c = i % 16;
        float v = 0.0f;
        #pragma unroll
        for (int ks = 0; ks < kKSplit; ++ks) v += smem.part()[ks * kMTiles + tile][r * 16 + c];
        const int t = tile * 16 + r, ex = expert_base + c;
        if (t < m && ex < e) part_logits[static_cast<int64_t>(t) * e + ex] = v;   // fp32 partial
    }
}

// Standalone-FE router CTA body (pipelined: chunk c+kStages-1 is issued while chunk
// c is consumed). `unit_idx` == blockIdx.x, `tid` == threadIdx.x, `sync` == __syncthreads.
template <int kMTiles, bool kTiny, typename Sync>
__device__ __forceinline__ void router_unit(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        float* __restrict__ logits,          // [k_split][m][e] workspace
        unsigned long long* stamps,
        int m, int h, int e, int unit_idx, int tid, uint8_t* dyn_smem, const Sync& sync) {
    using namespace nvcuda;
    using Cfg = RouterCfg<kMTiles, kTiny>;
    constexpr int kStages = Cfg::kStages;
    RouterSmem<kMTiles, kTiny> smem {dyn_smem};
    const int warp = tid >> 5;
    const int num_chunks = h / kChunkK / Cfg::kKSplitCTAs;

    router_zero_padding(smem, m, tid);
    sync();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int c = 0; c < kStages - 1; ++c) {
        if (c < num_chunks) router_issue_chunk(smem, hidden, router_weight, m, h, unit_idx, c, tid);
        cp_async_commit();
    }
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        if (chunk + kStages - 1 < num_chunks)
            router_issue_chunk(smem, hidden, router_weight, m, h, unit_idx, chunk + kStages - 1, tid);
        cp_async_commit();
        cp_async_wait<kStages - 1>();     // chunk `chunk` has landed for this thread
        sync();                           // ... and for every thread
        if (chunk == 0) stamp(stamps, 1);
        router_mma_chunk(smem, chunk % kStages, warp, acc);
        sync();                           // stage `st` may be refilled next iteration
    }
    stamp(stamps, 2);
    router_reduce_store(smem, logits, m, e, unit_idx, tid, acc, sync);
}

// Tiny-M top-k: same selection rule (largest value, smallest expert id on ties) and
// same fp32 softmax math as the legacy per-warp top-k, restructured for latency:
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

// Called by the whole 256-thread crew: the kKSplit partial logits of every expert
// are fetched by all threads (<= 2 experts per thread, one L2 round trip), summed
// in the legacy order and parked as keys in `key_s` (kMaxExperts u32); warp 0 then
// selects. Threads of warps 1..7 return after the key fill (one `sync` inside).
template <int kKSplit, int kPerLane, int kTopK, typename Sync>
__device__ __forceinline__ void topk_softmax_token_tiny(
        const float* __restrict__ logits, int64_t* __restrict__ topk_idx,
        float* __restrict__ topk_weights, unsigned long long* stamps, int t, int m, int e,
        int tid, uint32_t* key_s, const Sync& sync) {
    const int lane = tid & 31;
    // All partial loads of this thread's experts are issued before the first sum
    // (one L2 round trip for the whole fetch); the per-expert sum order is the legacy one.
    constexpr int kExPerThread = (kPerLane * 32 + kThreads - 1) / kThreads;
    float p[kExPerThread][kKSplit];
    #pragma unroll
    for (int i = 0; i < kExPerThread; ++i) {
        const int ex = tid + i * kThreads;
        #pragma unroll
        for (int ks = 0; ks < kKSplit; ++ks)
            p[i][ks] = (ex < e) ? __ldcg(logits + (static_cast<int64_t>(ks) * m + t) * e + ex) : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < kExPerThread; ++i) {
        const int ex = tid + i * kThreads;
        if (ex < kPerLane * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int ks = 0; ks < kKSplit; ++ks) acc += p[i][ks];      // same order as legacy
            key_s[ex] = topk_key(ex < e ? round_bf16(acc) : -INFINITY, ex);
        }
    }
    sync();
    stamp(stamps, 4);
    if (tid >= 32) return;
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

}  // namespace deep_gemm::fable_fe
