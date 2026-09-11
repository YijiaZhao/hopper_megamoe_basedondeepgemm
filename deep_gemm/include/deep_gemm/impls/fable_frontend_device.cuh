#pragma once
// Fable frontend device code (router WMMA, tiny-M top-k + softmax, activation
// quantisation), shared by the standalone kernel (csrc/fable_frontend.cu) and the
// SM90 fused MegaMoE kernel with the frontend fused in (kFuseFE, see
// docs/fe_into_mega_design.md). Kernel-agnostic: callers pass the thread index
// inside the 256-thread crew (`tid`), a 256-thread barrier (`sync`) and the smem
// regions; the math (WMMA fragment order, K-split partial order, bf16 rounding,
// gating fp32 math, quantisation) is exactly the standalone frontend's, so both
// paths produce bit-identical outputs.
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
// Optional per-CTA phase stamps (standalone FE, DG_FE_STAMPS=1): [blockIdx][8] u64 ns.
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
// kTiny (DG_FE_TINYM, m <= 16 only): 3 stages instead of 8. With 8 stages the
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
        int m, int h, int unit_idx, int chunk, int tid, int w_hint, uint64_t w_policy) {
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
        const __nv_bfloat16* src = router_weight + static_cast<int64_t>(expert_base + r) * h + k0 + c;
        if (w_hint) cp_async_16_hint(&smem.w(st)[r][c], src, w_policy);
        else cp_async_16(&smem.w(st)[r][c], src);
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
        int m, int h, int e, int w_hint, int unit_idx, int tid, uint8_t* dyn_smem, const Sync& sync) {
    using namespace nvcuda;
    using Cfg = RouterCfg<kMTiles, kTiny>;
    constexpr int kStages = Cfg::kStages;
    RouterSmem<kMTiles, kTiny> smem {dyn_smem};
    const int warp = tid >> 5;
    const int num_chunks = h / kChunkK / Cfg::kKSplitCTAs;
    const uint64_t w_policy = w_hint ? l2_evict_last_policy() : 0ull;

    router_zero_padding(smem, m, tid);
    sync();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int c = 0; c < kStages - 1; ++c) {
        if (c < num_chunks) router_issue_chunk(smem, hidden, router_weight, m, h, unit_idx, c, tid, w_hint, w_policy);
        cp_async_commit();
    }
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        if (chunk + kStages - 1 < num_chunks)
            router_issue_chunk(smem, hidden, router_weight, m, h, unit_idx, chunk + kStages - 1, tid, w_hint, w_policy);
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

// Fused-path split (kTiny only, num_chunks <= kStages so every chunk owns a stage):
// issue every chunk of a unit up front (caller commits, waits and syncs once for all
// its units), then compute. Same per-warp accumulation sequence as `router_unit`.
template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_unit_issue_all(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        int m, int h, int unit_idx, int tid, uint8_t* dyn_smem) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    RouterSmem<kMTiles, kTiny> smem {dyn_smem};
    const int num_chunks = h / kChunkK / Cfg::kKSplitCTAs;
    router_zero_padding(smem, m, tid);
    for (int chunk = 0; chunk < num_chunks; ++chunk)
        router_issue_chunk(smem, hidden, router_weight, m, h, unit_idx, chunk, tid, 0, 0ull);
}
template <int kMTiles, bool kTiny, typename Sync>
__device__ __forceinline__ void router_unit_compute(
        float* __restrict__ logits, int m, int h, int e, int unit_idx, int tid, uint8_t* dyn_smem,
        const Sync& sync) {
    using namespace nvcuda;
    using Cfg = RouterCfg<kMTiles, kTiny>;
    RouterSmem<kMTiles, kTiny> smem {dyn_smem};
    const int warp = tid >> 5;
    const int num_chunks = h / kChunkK / Cfg::kKSplitCTAs;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int chunk = 0; chunk < num_chunks; ++chunk)
        router_mma_chunk(smem, chunk % Cfg::kStages, warp, acc);
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
    for (int ex = tid; ex < kPerLane * 32; ex += kThreads) {
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

// ----------------------------------------------------------------- quantisation
// mode 0: FP8 E4M3 per K128 group (16 lanes x 8 values), sf = amax / 448.
// mode 1: INT8 whole row, sf = amax / 127 replicated into every K128 slot.
// 256-thread crew; `smem_warp_max` = 8 floats (mode 1 CTA-wide amax).
template <int kMode, typename Sync>
__device__ __forceinline__ void quant_role(
        const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
        float* __restrict__ x_sf, int token, int h, int tid, float* smem_warp_max, const Sync& sync) {
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* xrow = hidden + static_cast<int64_t>(token) * h;
    uint8_t* qrow = x_bytes + static_cast<int64_t>(token) * h;
    float* sfrow = x_sf + static_cast<int64_t>(token) * (h / 128);
    float row_scale = 0.0f;
    if constexpr (kMode == 1) {
        float local = 0.0f;
        for (int k0 = tid * 8; k0 < h; k0 += kThreads * 8) {
            float x[8]; unpack8(*reinterpret_cast<const uint4*>(xrow + k0), x);
            #pragma unroll
            for (int j = 0; j < 8; ++j) local = fmaxf(local, fabsf(x[j]));
        }
        local = warp_max(local);
        if (lane == 0) smem_warp_max[warp] = local;
        sync();
        float v = lane < kThreads / 32 ? smem_warp_max[lane] : 0.0f;
        v = warp_max(v);
        row_scale = fmaxf(v * (1.0f / 127.0f), 1.0e-30f);
        for (int g = tid; g < h / 128; g += kThreads) sfrow[g] = row_scale;
    }
    // Every (warp, iteration) covers 256 K = two K128 groups; half-warp = one group.
    for (int k0 = tid * 8; k0 < h; k0 += kThreads * 8) {
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

}  // namespace deep_gemm::fable_fe
