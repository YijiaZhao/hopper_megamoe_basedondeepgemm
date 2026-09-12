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
#include <deep_gemm/impls/fable_frontend_device.cuh>
#include <deep_gemm/impls/fable_cc_select.cuh>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

namespace {
// Device code (router WMMA, tiny top-k, quantisation) lives in
// deep_gemm/include/deep_gemm/impls/fable_frontend_device.cuh so that the fused
// MegaMoE kernel (DG_FP4_FUSE_FE) runs the very same math; this file keeps the
// launch, the CTA roles and the legacy per-warp top-k.
using namespace deep_gemm::fable_fe;

// full-K scheme limits (the shared constants / helpers -- kExpertsPerCTA, kThreads, kMaxTopK,
// kMaxExperts, warp reductions, cvt_e4m3x2, globaltimer_ns, stamp -- come from the header)
constexpr int kFullKStages = 12;                 // K chunks in flight per CTA (h <= 3072 -> all of them)
constexpr int kCandSlots = 8;                    // candidate keys per (token, expert group) when epc <= 8 (16 otherwise)
constexpr int kMaxCandSlots = 16;                // experts per CTA (group) <= 16 (one WMMA B tile)
constexpr int kMaxSlotsPerLane = 48;             // merger warp: keys per lane (groups x slots <= 1536)
constexpr int kMaxRouterCTAs = 128;              // groups x k_parts (partials / flags scratch)
// workspace (after the 256 B ticket area): [0, 64 KB) candidate keys, [128 KB, 256 KB) fp32
// K-part partials [cta][16][16], [256 KB, 256 KB + 512 B) K-part flags [cta]
constexpr size_t kFullKPartialsOff = 128 * 1024;
constexpr size_t kFullKFlagsOff = 256 * 1024;
// Stamp slots (DG_FE_STAMPS=1, [blockIdx][8] u64 ns; the 96-grid legend is in the header):
// full-K router CTA: 0 start / 1 chunk0 landed / 2 mma done / 3 keys written /
//                    4 quant done (CTAs t < m only) / 5 all chunks issued /
//                    6 %smid / 7 prologue done (before issue)
// full-K merger CTA (token 0's warp): 0 start / 1 first CTA seen / 2 last CTA
//                    seen / 3 merge done / 4 top-k written
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
struct CtaSync {
    __device__ __forceinline__ void operator()() const { __syncthreads(); }
};

// Router CTA: unit == blockIdx.x (16 experts x one K-part), dynamic smem.
template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_role(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        float* __restrict__ logits,          // [m, e] workspace
        unsigned long long* stamps,
        int m, int h, int e, int w_hint) {
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    router_unit<kMTiles, kTiny>(hidden, router_weight, logits, stamps, m, h, e, w_hint,
                                static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x), dyn_smem, CtaSync{});
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

// Tiny-M top-k CTA wrapper (static smem for the candidate keys).
using fable_cc::topk_key_value;
using fable_cc::topk_key_index;
template <int kKSplit, int kPerLane, int kTopK>
__device__ __forceinline__ void topk_tiny_cta(
        const float* __restrict__ logits, int64_t* __restrict__ topk_idx,
        float* __restrict__ topk_weights, unsigned long long* stamps, int t, int m, int e) {
    __shared__ uint32_t key_s[kMaxExperts];
    topk_softmax_token_tiny<kKSplit, kPerLane, kTopK>(logits, topk_idx, topk_weights, stamps, t, m, e,
                                                      static_cast<int>(threadIdx.x), key_s, CtaSync{});
}

// ----------------------------------------------------------------- quant CTA
// mode 0: FP8 E4M3 per K128 group (16 lanes x 8 values), sf = amax / 448.
// mode 1: INT8 whole row, sf = amax / 127 replicated into every K128 slot.
// quant_role_range: the quantisation of one token by threads tid = 0..nthreads-1 (a multiple of
// 32) of the CTA; bar_id 0 = __syncthreads (whole CTA), else a named barrier over nthreads (a
// thread subset, e.g. the merger CTA's idle warps while its warps 0..7 poll the keys).
template <int kMode>
__device__ __forceinline__ void quant_role_range(
        const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
        float* __restrict__ x_sf, int token, int h, int tid, int nthreads, int bar_id) {
    __shared__ float smem_warp_max[32];
    const int warp = tid >> 5, lane = tid & 31;
    auto sync = [&]() {
        if (bar_id == 0) __syncthreads();
        else asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(nthreads) : "memory");
    };
    const __nv_bfloat16* xrow = hidden + static_cast<int64_t>(token) * h;
    uint8_t* qrow = x_bytes + static_cast<int64_t>(token) * h;
    float* sfrow = x_sf + static_cast<int64_t>(token) * (h / 128);
    float row_scale = 0.0f;
    if constexpr (kMode == 1) {
        float local = 0.0f;
        for (int k0 = tid * 8; k0 < h; k0 += nthreads * 8) {
            float x[8]; unpack8(*reinterpret_cast<const uint4*>(xrow + k0), x);
            #pragma unroll
            for (int j = 0; j < 8; ++j) local = fmaxf(local, fabsf(x[j]));
        }
        local = warp_max(local);
        if (lane == 0) smem_warp_max[warp] = local;
        sync();
        float v = lane < nthreads / 32 ? smem_warp_max[lane] : 0.0f;
        v = warp_max(v);
        row_scale = fmaxf(v * (1.0f / 127.0f), 1.0e-30f);
        for (int g = tid; g < h / 128; g += nthreads) sfrow[g] = row_scale;
    }
    // Every (warp, iteration) covers 256 K = two K128 groups; half-warp = one group.
    for (int k0 = tid * 8; k0 < h; k0 += nthreads * 8) {
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

// Quant CTA wrapper: the whole CTA (kBlock threads, __syncthreads) quantises one token.
template <int kMode, int kBlock = kThreads>
__device__ __forceinline__ void quant_cta(
        const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
        float* __restrict__ x_sf, int token, int h) {
    quant_role_range<kMode>(hidden, x_bytes, x_sf, token, h, threadIdx.x, kBlock, 0);
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
//     front as TMA 1D bulk copies into row-contiguous smem (stride K-part + 16): 1-3
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
// over its K-part, stride (K-part + 16) bf16 (full K: 6176 B; 32 B aligned for WMMA, 2-way bank
// spread), so a TMA piece of a row is one contiguous copy. 16 rows of tail after the
// weight rows keep the (discarded) B-tile rows inside the allocation.
__host__ __device__ constexpr int fullk_smem_bytes(int m, int kpart_len) {
    return (m + 16) * (kpart_len + 16) * 2 + (kThreads / 32) * 256 * 4;
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
// smem_u32 / mbar_init come from the shared header.
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
using fable_cc::warp_max_u32_redux;
__device__ __forceinline__ uint32_t ld_cv_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
// 16 B volatile poll (4 consecutive key slots). Measured on H20 (cc path, kernel end): ld.cv scalar
// 5.12-5.38 us, ld.relaxed.gpu 5.89-6.14, ld.global.cg 6.14-6.40 -> .cv stays; 4 slots per load
// cuts a poll round from 20 to 5 loads per lane.
__device__ __forceinline__ uint4 ld_cv_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
// K-part exchange (k_parts > 1): CTA kp > 0 of a group publishes its fp32 partial
// logits [16 rows][16 experts] + a release flag; the kp == 0 CTA acquires every
// flag, sums own part 0 + parts 1..K-1 in that fixed order and emits the keys.
// Returns the final fp32 logit for this thread's (r, c) or NaN-free garbage when
// r >= m / c >= n_exp (the caller masks). Flags are reset by the consumer.
__device__ __forceinline__ float kpart_combine(float v, float* partials, int* flags, int k_parts, int kp, int r, int c,
                                               unsigned long long* stamps) {
    if (k_parts == 1) return v;
    const int cta = blockIdx.x;
    if (kp > 0) {
        partials[static_cast<int64_t>(cta) * 256 + r * 16 + c] = v;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) st_release_gpu(flags + cta, 1);
        return v;
    }
    if (threadIdx.x == 0) {
        for (int j = 1; j < k_parts; ++j) {
            while (ld_acquire_gpu(flags + cta + j) == 0) { }
            flags[cta + j] = 0;
        }
    }
    __syncthreads();
    stamp(stamps, 4);      // all K-part partials seen (kp == 0 CTAs; slot 4 = quant done on CTAs t < m otherwise)
    for (int j = 1; j < k_parts; ++j) v += __ldcg(partials + static_cast<int64_t>(cta + j) * 256 + r * 16 + c);
    return v;
}
__device__ __forceinline__ void emit_keys(float v, uint32_t* cand, int m, int n_exp, int expert_base, int groups,
                                          int group, int cand_slots) {
    const int r = threadIdx.x / 16, c = threadIdx.x % 16;
    if (r < m && c < cand_slots) {
        const int ex = expert_base + c;
        const uint32_t key = c < n_exp ? topk_key(round_bf16(v), ex) : 1u;
        cand[(static_cast<int64_t>(r) * groups + group) * cand_slots + c] = key;
    }
}

template <int kMode>
__device__ __forceinline__ void router_role_fullk(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint32_t* __restrict__ cand,           // [m][groups][cand_slots] keys
        float* partials, int* flags,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        unsigned long long* stamps,
        int m, int h, int e, int epc, int k_parts, int groups, int cand_slots, int w_hint) {
    using namespace nvcuda;
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    const int kpart_len = h / k_parts, ld = kpart_len + 16;          // 32 B aligned row stride
    auto row_s = [&](int r) { return reinterpret_cast<__nv_bfloat16*>(dyn_smem) + static_cast<int>(r) * ld; };
    float (*part_s)[256] = reinterpret_cast<float (*)[256]>(dyn_smem + (m + 16) * ld * 2);
    constexpr int kKSplit = kThreads / 32;                      // 8 warps = 8 K-slices per chunk
    constexpr int kKPerWarp = kChunkK / kKSplit;                // 32
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.x / k_parts, kp = blockIdx.x % k_parts;
    const int expert_base = group * epc;
    const int n_exp = min(epc, e - expert_base);
    const int num_chunks = kpart_len / kChunkK;
    const int n_rows = m + n_exp;
    const int kbase = kp * kpart_len;
    // TMA pieces per row: the bulk-copy queue is shallow (72 x 512 B copies per CTA
    // took ~2.5 us to *issue* on H20), keep the copy count <= ~24-32.
    int parts = max(1, min(3, 24 / n_rows));
    while (num_chunks % parts) --parts;
    const int chunks_per_part = num_chunks / parts;
    const uint32_t piece_bytes = static_cast<uint32_t>(chunks_per_part * kChunkK * 2);
    const uint64_t w_policy = w_hint ? l2_evict_last_policy() : 0ull;
    __shared__ __align__(8) uint64_t piece_bar[3];
    if (threadIdx.x < parts) mbar_init(&piece_bar[threadIdx.x], 1);
    mbar_fence_init();
    __syncthreads();
    stamp(stamps, 7);
    if (warp == 0) {
        if (lane < parts) mbar_arrive_expect_tx(&piece_bar[lane], static_cast<uint32_t>(n_rows) * piece_bytes);
        __syncwarp();
        for (int i = lane; i < parts * n_rows; i += 32) {
            const int pc = i / n_rows, r = i % n_rows, k0 = pc * chunks_per_part * kChunkK;
            const __nv_bfloat16* src = r < m ? hidden + static_cast<int64_t>(r) * h + kbase + k0
                                             : router_weight + static_cast<int64_t>(expert_base + r - m) * h + kbase + k0;
            if (w_hint && r >= m) tma_bulk_g2s_hint(row_s(r) + k0, src, piece_bytes, &piece_bar[pc], w_policy);
            else tma_bulk_g2s(row_s(r) + k0, src, piece_bytes, &piece_bar[pc]);
        }
    }
    stamp(stamps, 5);
    if (static_cast<int>(blockIdx.x) < m) {
        quant_cta<kMode>(hidden, x_bytes, x_sf, blockIdx.x, h);
        stamp(stamps, 4);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        if (chunk % chunks_per_part == 0) {
            mbar_wait_parity(&piece_bar[chunk / chunks_per_part], 0u);
            if (chunk == 0) stamp(stamps, 1);
        }
        #pragma unroll
        for (int kk = 0; kk < kKPerWarp; kk += 16) {
            const int k = chunk * kChunkK + warp * kKPerWarp + kk;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(a, row_s(0) + k, ld);
            wmma::load_matrix_sync(b, row_s(m) + k, ld);   // B(k, n) = w_s[n][k]
            wmma::mma_sync(acc, a, b, acc);
        }
    }
    wmma::store_matrix_sync(part_s[warp], acc, 16, wmma::mem_row_major);
    __syncthreads();
    stamp(stamps, 2);
    const int r = threadIdx.x / 16, c = threadIdx.x % 16;
    float v = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < kKSplit; ++ks) v += part_s[ks][r * 16 + c];
    v = kpart_combine(v, partials, flags, k_parts, kp, r, c, stamps);
    if (kp == 0) emit_keys(v, cand, m, n_exp, expert_base, groups, group, cand_slots);
}

// CUDA-core variant of the full-K router (DG_FE_TINYM_MMA=fma): no smem ring, no
// TMA, no WMMA fragments, no per-stage __syncthreads. Thread t owns the 16 B
// vectors t and t + 256 (h = 3072 -> 384 vectors per row, so threads < 128 own two)
// of every row; it issues ALL its expert weight vectors up front (<= 8 experts x 2
// = 16 x ld.global.nc.L1::no_allocate, each thread >= 5 loads in flight, the CTA
// ~36 KB) and then, per token row, loads the row's 2 activation vectors (L2-hot,
// every CTA reads the same rows) and FMAs. Accumulation order (fixed, documented):
// per thread and expert, acc = fma chain over the 8 bf16 pairs of vector t in
// element order, continued over the 8 pairs of vector t + 256; then the 32 lane
// partials are summed by the xor-butterfly (16, 8, 4, 2, 1); then the 8 warp
// partials are summed in warp order 0..7 through smem; one bf16 rounding.
__device__ __forceinline__ uint4 ld_nc_na_16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float dot8_bf16(const uint4& w, const uint4& x, float acc) {
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&x);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 a = __bfloat1622float2(wp[j]), b = __bfloat1622float2(xp[j]);
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
    }
    return acc;
}
template <int kMode, int kMaxExp, bool kTwoVec>
__device__ __forceinline__ void router_role_fullk_fma(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint32_t* __restrict__ cand, float* partials, int* flags,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        unsigned long long* stamps,
        int m, int h, int e, int epc, int k_parts, int groups, int cand_slots) {
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    float (*part_s)[kMaxCandSlots][16] = reinterpret_cast<float (*)[kMaxCandSlots][16]>(dyn_smem);   // [warp][expert][row]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.x / k_parts, kp = blockIdx.x % k_parts;
    const int expert_base = group * epc;
    const int n_exp = min(epc, e - expert_base);
    const int kpart_len = h / k_parts, nvec = kpart_len / 8, kbase = kp * kpart_len;
    const int v0 = threadIdx.x, v1 = threadIdx.x + kThreads;
    const bool has0 = v0 < nvec, has1 = kTwoVec && v1 < nvec;
    stamp(stamps, 7);
    uint4 w0[kMaxExp], w1[kTwoVec ? kMaxExp : 1];
    #pragma unroll
    for (int i = 0; i < kMaxExp; ++i) {
        w0[i] = make_uint4(0u, 0u, 0u, 0u);
        if (kTwoVec) w1[i] = w0[i];
        if (i < n_exp) {
            const __nv_bfloat16* wr = router_weight + static_cast<int64_t>(expert_base + i) * h + kbase;
            if (has0) w0[i] = ld_nc_na_16(wr + v0 * 8);
            if (has1) w1[i] = ld_nc_na_16(wr + v1 * 8);
        }
    }
    stamp(stamps, 5);
    if (static_cast<int>(blockIdx.x) < m) {
        quant_cta<kMode>(hidden, x_bytes, x_sf, blockIdx.x, h);
        stamp(stamps, 4);
    }
    for (int r = 0; r < m; ++r) {
        const __nv_bfloat16* xr = hidden + static_cast<int64_t>(r) * h + kbase;
        const uint4 x0 = has0 ? ld_nc_na_16(xr + v0 * 8) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 x1 = has1 ? ld_nc_na_16(xr + v1 * 8) : make_uint4(0u, 0u, 0u, 0u);
        #pragma unroll
        for (int i = 0; i < kMaxExp; ++i) {
            float acc = dot8_bf16(w0[i], x0, 0.0f);
            if (kTwoVec) acc = dot8_bf16(w1[i], x1, acc);
            if (r == 0 && i == 0) stamp(stamps, 1);
            acc = warp_sum(acc);
            if (lane == 0) part_s[warp][i][r] = acc;
        }
    }
    __syncthreads();
    stamp(stamps, 2);
    const int r = threadIdx.x / 16, c = threadIdx.x % 16;
    float v = 0.0f;
    if (c < kMaxExp) {
        #pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) v += part_s[w][c][r];
    }
    v = kpart_combine(v, partials, flags, k_parts, kp, r, c, stamps);
    if (kp == 0) emit_keys(v, cand, m, n_exp, expert_base, groups, group, cand_slots);
}


// ---------------------------------------------- CUDA-core K-split router (mma = cc)
// DG_FE_TINYM_MMA=cc | cc6 (full-K grid, k_parts 1, m <= 2, h = 3072, epc <= 5): CTA of
// 5 x kKS warps (kKS = 4 -> 640 threads, cc6: 6 -> 960). Warp (slot s, ks) owns expert
// expert_base + s over K-part ks (3072 / kKS elements); lane l owns the 16 B chunks
// l, l + 32, ... of that part (12 / kKS chunks). The kernel's first instructions issue the
// m activation chunks (L2-hot) and then every weight chunk of the CTA straight into
// registers (ld.global.nc.L1::no_allocate.v4; no smem, no TMA, no barrier before the
// FMAs). Microbench (csrc/router_cc_bench.cu, H20-3e, L2 flushed): the issue itself is
// back-pressured by the SM's outstanding-request capacity (even 2 loads/lane take ~1.5 us
// to issue), so more warps with fewer chunks each finish sooner: 8 warps x 12 chunks ->
// all logits at 2.9 us, 20 x 3 -> 2.05, 30 x 2 -> 2.02 (rows 1; rows 2: 2.66 / 2.56);
// 2 experts/warp spills at 255 regs (8+ us); cp.async.bulk into smem is slower (2.75).
// Accumulation order (fixed): per lane, fma chain over the 8 bf16 of chunk c in element
// order, c ascending (K = ks * K-part + 256 c + 8 lane + i); xor butterfly (16, 8, 4, 2, 1)
// over the lanes; K-part partials summed ks = 0..kKS-1 through smem; one bf16 rounding.
// Quantisation of the m rows runs on the merger CTA (idle until the first keys land) instead
// of router CTAs 0..m-1, so no router CTA carries extra loads. Hand-off unchanged: keys into
// cand[token][group][8] (sentinel 1 for slots >= n_exp), the merger polls the slots.
constexpr int kMergerDeferBit = 16;      // w_hint flag: merger defers the top-8 merge to one pass (DG_FE_MERGER_DEFER)
constexpr int kTicketMergeBit = 32;      // w_hint flag (cc): no polling merger; the LAST router CTA (atomic ticket) merges (DG_FE_CC_MERGE=ticket)
constexpr int kSelectReduxBit = 64;      // w_hint flag (cc ticket): top-8 by 8 redux rounds over register keys instead of insertion + merge8 (DG_FE_CC_SELECT=redux)
constexpr int kMergerFlagsMask = 15;     // w_hint bits below the merger flags
constexpr int kCCMaxM = 2;
constexpr int kCCH = 3072;
// kCC encodes the cc CTA shape: slots * 8 + K-split (44 = 5 experts x 4 warps = 640 threads, 46 = 5 x 6 = 960,
// 36 = 4 x 4 = 512 threads for a 96 + 1 CTA grid)
__host__ __device__ constexpr int cc_slots(int cc) { return cc / 8; }
__host__ __device__ constexpr int cc_ks(int cc) { return cc % 8; }
__host__ __device__ constexpr int cc_block(int cc) { return cc > 0 ? cc_slots(cc) * cc_ks(cc) * 32 : kThreads; }
constexpr int kTicketRelaxedBit = 128;
constexpr int kSelectInMegaBit = 512;    // w_hint flag (cc, DG_FE_SELECT_IN_MEGA=1, mma 8): router CTAs end after storing their keys; the fused MegaMoE prologue selects (deep_gemm/impls/fable_cc_select.cuh)
constexpr int kKeysCompactBit = 256;     // w_hint flag (cc ticket): keys in a compact [token][e] array (DG_FE_CC_KEYS=compact, default when e == 384)
constexpr int kSelectPrunedBit = 1024;   // w_hint flag (cc ticket, compact keys): two-level threshold select (DG_FE_CC_SELECT=pruned, fable_cc::select_pruned)
constexpr size_t kCCKeysOff = 64 * 1024; // workspace offset of the compact keys ([2][512] u32 max)
constexpr int kCCW8Pad = 16;             // fp8 router weight row: h bytes e4m3 + fp32 row scale + pad (row stride h + 16)
// globaltimer read predicated on `dep`: cannot be scheduled before the load producing it lands
__device__ __forceinline__ unsigned long long globaltimer_after(uint32_t dep) {
    unsigned long long t;
    asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %1, 0x7fffffff;\n @p mov.u64 %0, %%globaltimer;\n @!p mov.u64 %0, 0;\n}"
                 : "=l"(t) : "r"(dep));
    return t;
}
// e4m3 x2 (low 16 bits of `v`) -> float2 via cvt.rn.f16x2.e4m3x2
__device__ __forceinline__ float2 e4m3x2_to_float2(uint32_t v) {
    uint32_t h2;
    asm("{\n .reg .b16 lo;\n mov.b16 lo, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(h2) : "h"(static_cast<unsigned short>(v & 0xFFFFu)));
    return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
// acc += <16 e4m3 of w (scaled later), 16 bf16 of x (two uint4)>
__device__ __forceinline__ float dot16_e4m3(const uint4& w, const uint4& x0, const uint4& x1, float acc) {
    const uint32_t wd[4] = {w.x, w.y, w.z, w.w};
    const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
    const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 a = e4m3x2_to_float2(wd[j]), b = e4m3x2_to_float2(wd[j] >> 16);
        const float2 xa = __bfloat1622float2(j < 2 ? xp0[2 * j] : xp1[2 * (j - 2)]);
        const float2 xb = __bfloat1622float2(j < 2 ? xp0[2 * j + 1] : xp1[2 * (j - 2) + 1]);
        acc = fmaf(a.x, xa.x, acc); acc = fmaf(a.y, xa.y, acc);
        acc = fmaf(b.x, xb.x, acc); acc = fmaf(b.y, xb.y, acc);
    }
    return acc;
}
// kW8 (DG_FE_ROUTER_WDTYPE=fp8 experiment, mma = ccfp8): router weights e4m3 with one fp32 scale per
// expert row, packed [e][h + 16] bytes (host: fable_router_weight_fp8); a lane's 16 B chunk holds 16 K
// elements, so kKS = 3 (K-part 1024 = 64 chunks = 2 per lane); the fp32 sum is scaled once at the end.
template <int kSlots, int kKS, bool kW8>
__device__ __forceinline__ void router_role_fullk_cc(
        const __nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ router_weight,
        uint32_t* __restrict__ cand, uint32_t* __restrict__ ckeys, unsigned long long* stamps,
        int m, int h, int e, int epc, int groups, int cand_slots, bool compact) {
    constexpr int kKPart = kCCH / kKS;
    constexpr int kChunks = kW8 ? kKPart / 16 / 32 : kKPart / 8 / 32;    // 16 B chunks per lane
    static_assert(kChunks * 32 * (kW8 ? 16 : 8) == kKPart, "K-part must split into whole 16 B chunks per lane");
    __shared__ float part_s[kSlots][kCCMaxM][kKS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = warp / kKS, ks = warp % kKS;
    const int group = blockIdx.x, expert_base = group * epc;
    const int n_exp = min(epc, e - expert_base);
    const bool active = slot < n_exp;
    constexpr int kEPC = kW8 ? 16 : 8;                                    // K elements per 16 B weight chunk
    const int kbase = ks * kKPart + lane * kEPC;                          // K element offset of this lane's chunk 0
    stamp(stamps, 7);
    uint4 xv[kCCMaxM][kChunks * (kW8 ? 2 : 1)], wv[kChunks];
    #pragma unroll
    for (int r = 0; r < kCCMaxM; ++r)
        #pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            if constexpr (kW8) {
                xv[r][2 * c] = xv[r][2 * c + 1] = make_uint4(0u, 0u, 0u, 0u);
                if (active && r < m) {
                    xv[r][2 * c] = ld_nc_na_16(hidden + static_cast<int64_t>(r) * h + kbase + 32 * kEPC * c);
                    xv[r][2 * c + 1] = ld_nc_na_16(hidden + static_cast<int64_t>(r) * h + kbase + 32 * kEPC * c + 8);
                }
            } else {
                xv[r][c] = make_uint4(0u, 0u, 0u, 0u);
                if (active && r < m) xv[r][c] = ld_nc_na_16(hidden + static_cast<int64_t>(r) * h + kbase + 32 * kEPC * c);
            }
        }
    const uint8_t* wr8 = reinterpret_cast<const uint8_t*>(router_weight) + static_cast<int64_t>(expert_base + slot) * (h + kCCW8Pad) + kbase;
    const __nv_bfloat16* wr = router_weight + static_cast<int64_t>(expert_base + slot) * h + kbase;
    #pragma unroll
    for (int c = 0; c < kChunks; ++c) {
        wv[c] = make_uint4(0u, 0u, 0u, 0u);
        if (active) wv[c] = kW8 ? ld_nc_na_16(wr8 + 32 * 16 * c) : ld_nc_na_16(wr + 32 * kEPC * c);
    }
    float wscale = 1.0f;
    if constexpr (kW8) { if (active) wscale = __ldg(reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(router_weight) + static_cast<int64_t>(expert_base + slot) * (h + kCCW8Pad) + h)); }
    stamp(stamps, 5);
    if (stamps != nullptr && threadIdx.x == 0) stamps[blockIdx.x * kStampSlots + 1] = globaltimer_after(wv[0].x);
    float acc[kCCMaxM];
    #pragma unroll
    for (int r = 0; r < kCCMaxM; ++r) acc[r] = 0.0f;
    #pragma unroll
    for (int c = 0; c < kChunks; ++c)
        #pragma unroll
        for (int r = 0; r < kCCMaxM; ++r) {
            if constexpr (kW8) acc[r] = dot16_e4m3(wv[c], xv[r][2 * c], xv[r][2 * c + 1], acc[r]);
            else acc[r] = dot8_bf16(wv[c], xv[r][c], acc[r]);
        }
    #pragma unroll
    for (int r = 0; r < kCCMaxM; ++r) {
        acc[r] = warp_sum(acc[r]) * wscale;
        if (lane == 0 && slot < kSlots) part_s[slot][r][ks] = acc[r];
    }
    __syncthreads();
    stamp(stamps, 2);
    if (static_cast<int>(threadIdx.x) < cand_slots * kCCMaxM) {
        const int r = threadIdx.x / cand_slots, c = threadIdx.x % cand_slots;
        if (r < m) {
            uint32_t key = 1u;
            if (c < n_exp) {
                float v = 0.0f;
                #pragma unroll
                for (int k = 0; k < kKS; ++k) v += part_s[c][r][k];
                key = topk_key(round_bf16(v), expert_base + c);
            }
            if (compact) { if (c < n_exp) ckeys[static_cast<int64_t>(r) * e + expert_base + c] = key; }
            else cand[(static_cast<int64_t>(r) * groups + group) * cand_slots + c] = key;
        }
    }
}

// ------------------------------------------------------- swapped-operand MMA
// DG_FE_TINYM_MMA=swapab: the router product is computed as
//   D[16 experts x 8 tokens] += A[16 experts x k16] (weights) . B[k16 x 8 tokens] (activations)
// with mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32, i.e. the experts sit on
// the MMA M dimension (one m16 tile = the CTA's 16 experts) and the tokens on N
// (rows pad to 8, not 16; m in 9..16 -> a second n8 tile). The A (weight)
// fragments are loaded straight from global memory into registers (no smem, no
// ldmatrix); the B (activation) rows are staged once in smem (cp.async, <= m rows x
// K-part) and read as one 16 B vector per lane and k32 block.
// Fragment / offset design (K permuted inside every k32 block, identically for A
// and B, so the dot product is unchanged): lane l = (g = l / 4, t = l % 4).
//   * A fragment of the m16n8k16 (PTX): {a0,a1} = A[g][2t, 2t+1], {a2,a3} = A[g+8][2t, 2t+1],
//     {a4,a5} = A[g][2t+8, 2t+9], {a6,a7} = A[g+8][2t+8, 2t+9]; B: {b0,b1} = B[2t, 2t+1][g],
//     {b2,b3} = B[2t+8, 2t+9][g]; D: {c0,c1} = D[g][2t, 2t+1], {c2,c3} = D[g+8][2t, 2t+1].
//   * a k32 block at K offset k0 is consumed as two k16 steps s = 0, 1. Lane (g, t)
//     loads ONE 16 B vector per row it owns: W[expert g][k0 + 8t .. k0 + 8t + 7] (wg),
//     W[expert g+8][same] (wg8), X[token g][same] (xb). 32-bit word w of such a
//     vector holds physical K elements k0 + 8t + 2w, +1. Step s uses words 2s
//     (as the logical pair {2t, 2t+1}) and 2s+1 (as the logical pair {2t+8, 2t+9}):
//     A regs = {wg.word[2s], wg8.word[2s], wg.word[2s+1], wg8.word[2s+1]},
//     B regs = {xb.word[2s], xb.word[2s+1]}. Logical k = 2t + i + 8p <-> physical
//     k0 + 8t + 4s + 2p + i is a bijection for each s and depends on (t, p, i) only,
//     so A and B agree on every product; both steps together cover the 32 K values.
//   * per warp the 8 rows g and 8 rows g+8 x 4 lanes t = 16 experts x 32 K per block:
//     4 consecutive lanes read 64 contiguous bytes of one weight row (two 32 B
//     sectors; the neighbouring 64 B of the 128 B line belong to another warp's
//     block of the same CTA).
// Global offsets are precomputed per lane once (row base + 8t), the block loop only
// adds k0. All kBlk weight vectors of the lane are issued up front (kBlk = K-part /
// 256 blocks per warp = 6 * kBlk k16 steps in flight; 3 -> 6 steps in the legacy 96 x 4
// grid, 12 -> 24 steps in the 78 full-K grid) before the first MMA.
// K ownership / accumulation order (fixed, documented): warp w owns K-slice
// [32w, 32w + 32) of every 256-wide chunk of its K-part, block j = chunk j; the
// warp accumulates its blocks in order j = 0..kBlk-1 (steps s = 0, 1 each) in ONE
// fp32 accumulator (the 16-product sum inside one m16n8k16 is the hardware order);
// the 8 warp partials are summed in warp order 0..7 through smem; then (legacy) the
// 4 K-part partials in order ks = 0..3 in the top-k CTA, or (full-K) k_parts in order
// 0..k_parts-1 in the part-0 CTA; one bf16 rounding of the logit. Deterministic, not
// bit-identical to the WMMA paths.
__device__ __forceinline__ void mma_m16n8k16_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// smem layout for the swapped path: x rows [n_tiles * 8][ld] bf16 (ld = kpart_len + 8:
// row stride 2 * ld B = 16 mod 128 -> the 8 rows g of a warp's 16 B reads hit 8 disjoint
// 4-bank groups, conflict-free), then part_s[8 warps][256] fp32.
__host__ __device__ constexpr int swapab_smem_bytes(int m, int kpart_len) {
    return ((m + 7) / 8) * 8 * (kpart_len + 8) * 2 + (kThreads / 32) * 256 * 4;
}
template <int kBlk>
struct SwapABRouter {
    uint4 wg[kBlk], wg8[kBlk];
    // Issue every weight vector of this lane (rows g / g+8 of the CTA's 16-expert tile
    // over the K-part; rows >= n_exp are zeros, never loaded) and the cp.async of the
    // activation rows into smem (caller commits / waits: cp_async_wait<0> + __syncthreads).
    // wlayout 0: row-major [e][h] weights -- a warp's block load touches 8 rows x 64 B
    // (8 half lines, 16 sectors). wlayout 1 (DG_FE_ROUTER_WLAYOUT=fragment, host-side
    // one-time permutation, fable_router_weight_fragment_layout): the same bytes stored
    // in A-fragment order [e/16 groups][h/32 blocks][half: rows g | g+8][lane][8 bf16], so
    // lane l reads element offset ((G * (h/32) + b) * 2 + half) * 256 + 8 l and one warp
    // instruction is ONE contiguous 512 B run = exactly 4 full 128 B lines (16 sectors,
    // 4 line requests instead of 8). Same values into the same registers -> identical
    // numerics. Requires the CTA's experts to be an aligned 16-group (expert_base % 16 == 0).
    __device__ __forceinline__ void issue(const __nv_bfloat16* __restrict__ router_weight,
                                          const __nv_bfloat16* __restrict__ hidden, __nv_bfloat16* x_s,
                                          int m, int h, int expert_base, int n_exp, int kbase, int kpart_len, int wlayout) {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
        const __nv_bfloat16* wrow;
        const __nv_bfloat16* wrow8;
        int64_t jstride;
        if (wlayout == 0) {
            const int64_t kofs = kbase + warp * 32 + t * 8;
            wrow = router_weight + static_cast<int64_t>(expert_base + g) * h + kofs;
            wrow8 = wrow + static_cast<int64_t>(8) * h;
            jstride = kChunkK;
        } else {
            const int64_t b0 = static_cast<int64_t>(expert_base / 16) * (h / 32) + kbase / 32 + warp;
            wrow = router_weight + b0 * 512 + lane * 8;
            wrow8 = wrow + 256;
            jstride = static_cast<int64_t>(8) * 512;             // next block of this warp = 8 blocks further
        }
        #pragma unroll
        for (int j = 0; j < kBlk; ++j) {
            wg[j] = g < n_exp ? ld_nc_na_16(wrow + j * jstride) : make_uint4(0u, 0u, 0u, 0u);
            wg8[j] = g + 8 < n_exp ? ld_nc_na_16(wrow8 + j * jstride) : make_uint4(0u, 0u, 0u, 0u);
        }
        const int ld = kpart_len + 8, nvec = kpart_len / 8, n_pad = ((m + 7) / 8) * 8;
        for (int i = threadIdx.x; i < m * nvec; i += kThreads) {
            const int r = i / nvec, c = (i % nvec) * 8;
            cp_async_16(x_s + r * ld + c, hidden + static_cast<int64_t>(r) * h + kbase + c);
        }
        cp_async_commit();
        for (int i = threadIdx.x; i < (n_pad - m) * nvec; i += kThreads) {      // zero the padding tokens
            const int r = m + i / nvec, c = (i % nvec) * 8;
            *reinterpret_cast<uint4*>(x_s + r * ld + c) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    // Blocks j = 0..kBlk-1 (steps s = 0, 1) into one fp32 accumulator per n8 tile, then
    // part_s[warp][token * 16 + expert] (the same [row][col] layout the WMMA paths store).
    __device__ __forceinline__ void compute(const __nv_bfloat16* x_s, float (*part_s)[256], int m, int kpart_len,
                                            unsigned long long* stamps) {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
        const int ld = kpart_len + 8, n_tiles = (m + 7) / 8;
        float acc[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < kBlk; ++j) {
            const int k0 = j * kChunkK + warp * 32 + t * 8;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                if (nt < n_tiles) {
                    const uint4 xb = *reinterpret_cast<const uint4*>(x_s + (nt * 8 + g) * ld + k0);
                    mma_m16n8k16_bf16(acc[nt], wg[j].x, wg8[j].x, wg[j].y, wg8[j].y, xb.x, xb.y);   // s = 0
                    mma_m16n8k16_bf16(acc[nt], wg[j].z, wg8[j].z, wg[j].w, wg8[j].w, xb.z, xb.w);   // s = 1
                }
            }
            if (j == 0) stamp(stamps, 1);
        }
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            if (nt < n_tiles) {
                const int tok = nt * 8 + 2 * t;
                part_s[warp][tok * 16 + g] = acc[nt][0];
                part_s[warp][(tok + 1) * 16 + g] = acc[nt][1];
                part_s[warp][tok * 16 + g + 8] = acc[nt][2];
                part_s[warp][(tok + 1) * 16 + g + 8] = acc[nt][3];
            }
        }
    }
};
// Legacy 96 x 4 grid (24 groups x 16 experts x 4 K-parts, m quant/top-k CTAs, ticket)
// with the swapped MMA: same CTA roles, partial logits [k_part][m][e] in fp32 as the
// WMMA router_role; only the load / MMA path differs. h / 256 / 4 == kBlk required.
template <int kBlk>
__device__ __forceinline__ void router_role_swapab(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        float* __restrict__ logits,          // [k_part][m][e] workspace partials
        unsigned long long* stamps,
        int m, int h, int e, int wlayout) {
    constexpr int kKSplitCTAs = RouterCfg<1, true>::kKSplitCTAs;
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    const int kpart_len = h / kKSplitCTAs;
    __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
    float (*part_s)[256] = reinterpret_cast<float (*)[256]>(dyn_smem + ((m + 7) / 8) * 8 * (kpart_len + 8) * 2);
    const int expert_base = (blockIdx.x / kKSplitCTAs) * kExpertsPerCTA;
    const int k_part = blockIdx.x % kKSplitCTAs;
    const int n_exp = min(kExpertsPerCTA, e - expert_base);
    float* part_logits = logits + static_cast<int64_t>(k_part) * m * e;
    SwapABRouter<kBlk> r;
    r.issue(router_weight, hidden, x_s, m, h, expert_base, n_exp, k_part * kpart_len, kpart_len, wlayout);
    stamp(stamps, 5);
    cp_async_wait<0>();
    __syncthreads();
    r.compute(x_s, part_s, m, kpart_len, stamps);
    __syncthreads();
    stamp(stamps, 2);
    const int row = threadIdx.x / 16, c = threadIdx.x % 16;
    float v = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < kThreads / 32; ++ks) v += part_s[ks][row * 16 + c];
    const int ex = expert_base + c;
    if (row < m && ex < e) part_logits[static_cast<int64_t>(row) * e + ex] = v;   // fp32 partial
}
// Full-K grid (DG_FE_TINYM_GRID=auto|N, K-parts 1|2|4) with the swapped MMA: the CTA's
// <= 16 experts are the m16 tile (rows >= n_exp zero), kBlk = K-part / 256 blocks per warp.
template <int kMode, int kBlk>
__device__ __forceinline__ void router_role_fullk_swapab(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint32_t* __restrict__ cand, float* partials, int* flags,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        unsigned long long* stamps,
        int m, int h, int e, int epc, int k_parts, int groups, int cand_slots, int wlayout) {
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    const int kpart_len = h / k_parts;
    __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
    float (*part_s)[256] = reinterpret_cast<float (*)[256]>(dyn_smem + ((m + 7) / 8) * 8 * (kpart_len + 8) * 2);
    const int group = blockIdx.x / k_parts, kp = blockIdx.x % k_parts;
    const int expert_base = group * epc;
    const int n_exp = min(epc, e - expert_base);
    stamp(stamps, 7);
    SwapABRouter<kBlk> r;
    r.issue(router_weight, hidden, x_s, m, h, expert_base, n_exp, kp * kpart_len, kpart_len, wlayout);
    stamp(stamps, 5);
    if (static_cast<int>(blockIdx.x) < m) {
        quant_cta<kMode>(hidden, x_bytes, x_sf, blockIdx.x, h);
        stamp(stamps, 4);
    }
    cp_async_wait<0>();
    __syncthreads();
    r.compute(x_s, part_s, m, kpart_len, stamps);
    __syncthreads();
    stamp(stamps, 2);
    const int row = threadIdx.x / 16, c = threadIdx.x % 16;
    float v = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < kThreads / 32; ++ks) v += part_s[ks][row * 16 + c];
    v = kpart_combine(v, partials, flags, k_parts, kp, row, c, stamps);
    if (kp == 0) emit_keys(v, cand, m, n_exp, expert_base, groups, group, cand_slots);
}

// Merger CTA: one warp per token (m <= 16 -> <= 2 tokens per warp). Lane l owns
// slots 4 (l + 32 i) .. + 3, i < kSlotsPerLane / 4, of cand[token] (16 B polls, a warp
// covers 512 B per load; a CTA's 8 keys land in 2 consecutive lanes). Register-lean streaming merge (no per-slot key array -> no
// local-memory spills): every poll round folds the newly-arrived keys into a
// per-lane sorted top-8 `loc[]` (insertion), then, unless the batch maximum cannot
// enter the running top-8 (early exit, the common case late in the stream), runs
// 8 rounds of one redux.sync max over (running key in lanes 0..7, loc[0]) with the
// winner popped -> new running top-8 in lanes 0..7, descending.
// merge8: running top-8 (lanes 0..7 of `run`, descending) <- top-8 of (run, every lane's sorted loc[])
using fable_cc::merge8;
using fable_cc::topk_finish;
using fable_cc::ld_cg_v4;
// Last-arriver merge (cc, DG_FE_CC_MERGE=ticket, default): called by the router CTA whose
// atom.acq_rel.gpu ticket was the last (every CTA: keys stored -> bar.sync -> thread 0 atomic;
// the release orders the CTA's key stores, the acquire on the last CTA orders its reads).
// Warp t < m: read token t's slots (5 x 16 B per lane, L2-hot) into registers, then 8 rounds of
// {per-lane max over its 20 keys (independent ops), one redux.sync max, the owner lane clears
// its copy} -> lanes 0..7 hold the top-8 descending (keys are unique: the expert id sits in the
// low bits; the sentinel 1 never wins while >= 8 real keys exist). ~0.3 us vs ~0.5-0.6 for a
// per-lane 20 x 8 sorted insertion chain + 8-round merge (serial dependent ALU chain in one warp).
// Stamps go to the merger CTA's slots (1 ticket won / 2 keys read / 3 merge done / 4 written).
template <int kTopK, int kSlotsPerLane>
__device__ __forceinline__ void last_arriver_topk(
        uint32_t* __restrict__ cand, int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        unsigned long long* mstamps, int m, int groups, int cand_slots, bool redux_sel, bool relaxed) {
    constexpr int kVecPerLane = kSlotsPerLane / 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nslots = groups * cand_slots;
    if (warp >= m) return;
    const int t = warp;
    uint32_t* base = cand + static_cast<int64_t>(t) * nslots;
    uint4 q[kVecPerLane];
    #pragma unroll
    for (int i = 0; i < kVecPerLane; ++i) {
        q[i] = make_uint4(0u, 0u, 0u, 0u);
        if (4 * (lane + 32 * i) < nslots) q[i] = ld_cg_v4(base + 4 * (lane + 32 * i));
    }
    if (mstamps != nullptr && t == 0 && lane == 0) mstamps[2] = globaltimer_after(q[0].x);
    if (relaxed) {           // relaxed ticket: a slot may not be visible yet -> data is the flag, re-read zeros
        while (true) {
            bool zero = false;
            #pragma unroll
            for (int i = 0; i < kVecPerLane; ++i)
                if (4 * (lane + 32 * i) < nslots && (q[i].x == 0u || q[i].y == 0u || q[i].z == 0u || q[i].w == 0u)) {
                    q[i] = ld_cg_v4(base + 4 * (lane + 32 * i));
                    zero |= q[i].x == 0u || q[i].y == 0u || q[i].z == 0u || q[i].w == 0u;
                }
            if (!__any_sync(0xffffffffu, zero)) break;
        }
    }
    uint32_t kv[kSlotsPerLane];
    #pragma unroll
    for (int i = 0; i < kVecPerLane; ++i) { kv[4 * i] = q[i].x; kv[4 * i + 1] = q[i].y; kv[4 * i + 2] = q[i].z; kv[4 * i + 3] = q[i].w; }
    uint32_t run = 0u;
    if (redux_sel) {
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            uint32_t tr[kSlotsPerLane];              // tree max (depth log2), not a serial chain
            #pragma unroll
            for (int i = 0; i < kSlotsPerLane; ++i) tr[i] = kv[i];
            #pragma unroll
            for (int st = 1; st < kSlotsPerLane; st *= 2)
                #pragma unroll
                for (int i = 0; i + st < kSlotsPerLane; i += 2 * st) tr[i] = max(tr[i], tr[i + st]);
            const uint32_t mx = tr[0];
            const uint32_t best = warp_max_u32_redux(mx);
            if (lane == k) run = best;
            if (mx == best) {
                #pragma unroll
                for (int i = 0; i < kSlotsPerLane; ++i) kv[i] = kv[i] == best ? 0u : kv[i];
            }
        }
    } else {                                   // per-lane sorted top-8 (insertion) + one 8-round merge
        uint32_t loc[kTopK];
        #pragma unroll
        for (int j = 0; j < kTopK; ++j) loc[j] = 0u;
        #pragma unroll
        for (int i = 0; i < kSlotsPerLane; ++i) {
            uint32_t k = kv[i];
            #pragma unroll
            for (int j = 0; j < kTopK; ++j) { const uint32_t lo = min(k, loc[j]); loc[j] = max(k, loc[j]); k = lo; }
        }
        merge8<kTopK>(run, loc, lane);
    }
    if (mstamps != nullptr && t == 0 && lane == 0) mstamps[3] = globaltimer_ns();
    topk_finish<kTopK>(run, lane, t, topk_idx, topk_weights);
    if (mstamps != nullptr && t == 0 && lane == 0) mstamps[4] = globaltimer_ns();
    #pragma unroll
    for (int i = 0; i < kVecPerLane; ++i)      // slots back to 0 (the polling merger mode relies on it)
        if (4 * (lane + 32 * i) < nslots) *reinterpret_cast<uint4*>(base + 4 * (lane + 32 * i)) = make_uint4(0u, 0u, 0u, 0u);
}
// Compact variant: token t's e (= 384) keys are contiguous -> 3 x 16 B per lane, insertion 12 x 8 + merge8
// (pruned: fable_cc::select_pruned, DG_FE_CC_SELECT=pruned, same result).
template <int kTopK>
__device__ __forceinline__ void last_arriver_topk_compact(
        uint32_t* __restrict__ ckeys, int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        unsigned long long* mstamps, int m, int e, bool relaxed, bool pruned = false) {
    constexpr int kVec = 3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= m) return;
    const int t = warp;
    uint32_t* base = ckeys + static_cast<int64_t>(t) * e;
    uint4 q[kVec];
    #pragma unroll
    for (int i = 0; i < kVec; ++i) q[i] = ld_cg_v4(base + 4 * (lane + 32 * i));
    if (mstamps != nullptr && t == 0 && lane == 0) mstamps[2] = globaltimer_after(q[0].x);
    if (relaxed) {
        while (true) {
            bool zero = false;
            #pragma unroll
            for (int i = 0; i < kVec; ++i)
                if (q[i].x == 0u || q[i].y == 0u || q[i].z == 0u || q[i].w == 0u) {
                    q[i] = ld_cg_v4(base + 4 * (lane + 32 * i));
                    zero |= q[i].x == 0u || q[i].y == 0u || q[i].z == 0u || q[i].w == 0u;
                }
            if (!__any_sync(0xffffffffu, zero)) break;
        }
    }
    uint32_t run = 0u;
    if (pruned) {
        fable_cc::select_pruned<kTopK>(q, lane, run);
    } else {
        uint32_t loc[kTopK];
        fable_cc::insert_keys<kTopK, kVec>(q, loc);
        merge8<kTopK>(run, loc, lane);
    }
    if (mstamps != nullptr && t == 0 && lane == 0) mstamps[3] = globaltimer_ns();
    topk_finish<kTopK>(run, lane, t, topk_idx, topk_weights);
    if (mstamps != nullptr && t == 0 && lane == 0) mstamps[4] = globaltimer_ns();
    #pragma unroll
    for (int i = 0; i < kVec; ++i) *reinterpret_cast<uint4*>(base + 4 * (lane + 32 * i)) = make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ int atom_add_acq_rel_gpu(int* p, int v) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
// defer (DG_FE_MERGER_DEFER, cc default 1): no per-batch merge; every lane keeps the sorted
// top-8 of ALL keys it has seen (a lane's top-8 is a superset of its contribution to the
// global top-8) and ONE 8-round merge runs after the last slot arrived. The router CTAs of the
// cc path finish within ~0.5 us of each other, so the streaming merge only lengthened the poll
// period (each round carried 8 redux rounds) and the lag behind the last key.
template <int kTopK, int kSlotsPerLane>
__device__ __forceinline__ void merger_role(
        uint32_t* __restrict__ cand, int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        unsigned long long* stamps, int m, int groups, int cand_slots, bool defer) {
    static_assert(kSlotsPerLane % 4 == 0, "16 B polls: 4 slots per load");
    constexpr int kVecPerLane = kSlotsPerLane / 4;   // lane l polls uint4 l + 32 i (slots 4 (l + 32 i) .. + 3)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nslots = groups * cand_slots;
    for (int t = warp; t < m; t += kThreads / 32) {
        uint32_t* base = cand + static_cast<int64_t>(t) * nslots;
        uint32_t pending = 0u;
        #pragma unroll
        for (int i = 0; i < kSlotsPerLane; ++i)
            if (4 * (lane + 32 * (i / 4)) + i % 4 < nslots) pending |= 1u << i;
        uint32_t run = 0u;                 // lanes 0..7: running top-8 (descending), else 0
        uint32_t loc[kTopK];               // this lane's sorted (desc) keys of the current batch
        #pragma unroll
        for (int j = 0; j < kTopK; ++j) loc[j] = 0u;
        bool first = true;
        while (true) {
            // issue every pending poll first (one L2 round trip per poll round), then fold in.
            // (Software-pipelining the next round's loads over the fold/merge was measured
            // slower: regs 95 -> 128, first-CTA-seen +1 us, kernel end 7.94 -> 8.70 us.)
            uint32_t v[kSlotsPerLane];
            #pragma unroll
            for (int i = 0; i < kVecPerLane; ++i) {
                uint4 q = make_uint4(0u, 0u, 0u, 0u);
                if (pending & (0xFu << (4 * i))) q = ld_cv_v4(base + 4 * (lane + 32 * i));
                v[4 * i] = q.x; v[4 * i + 1] = q.y; v[4 * i + 2] = q.z; v[4 * i + 3] = q.w;
            }
            #pragma unroll
            for (int i = 0; i < kSlotsPerLane; ++i)
                if (!(pending & (1u << i))) v[i] = 0u;
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
                if (!defer) {
                    const uint32_t batch_max = warp_max_u32_redux(loc[0]);
                    const uint32_t run8 = __shfl_sync(0xffffffffu, run, kTopK - 1);   // current 8th best (0 if < 8 yet)
                    if (batch_max > run8) merge8<kTopK>(run, loc, lane);
                    #pragma unroll
                    for (int j = 0; j < kTopK; ++j) loc[j] = 0u;   // losers can never re-enter
                }
            }
            if (__all_sync(0xffffffffu, pending == 0u)) break;
        }
        if (defer) merge8<kTopK>(run, loc, lane);
        if (t == 0) stamp(stamps, 3);
        topk_finish<kTopK>(run, lane, t, topk_idx, topk_weights);
        if (t == 0) stamp(stamps, 4);
        // reset the slots for the next launch (every writer has been consumed)
        #pragma unroll
        for (int i = 0; i < kSlotsPerLane; ++i)
            if (4 * (lane + 32 * (i / 4)) + i % 4 < nslots) base[4 * (lane + 32 * (i / 4)) + i % 4] = 0u;
    }
}

// kTiny: <= 85 regs/thread so 3 CTAs (59 KB smem each) fit per SM -> single wave.
// kFullK: <= 128 regs (merger warp holds 32 keys), 2 CTAs/SM cap; grid <= SM count anyway.
// kFma: 16 x uint4 weight vectors live in registers -> 1 CTA/SM bound (255 regs), no spills.
// kCC (> 0 = K-split factor): CUDA-core router role, cc_block(kCC) threads, 1 CTA/SM.
template <int kMTiles, int kMode, bool kTiny, bool kFullK, bool kFma, int kSwapBlk, int kCC = 0, bool kW8 = false>
__global__ void __launch_bounds__(cc_block(kCC), (kFma || kCC > 0) ? 1 : (kFullK ? 2 : (kTiny ? 3 : 1))) router_quant_topk_kernel(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        int* __restrict__ ticket, float* __restrict__ logits,
        unsigned long long* stamps,
        int m, int h, int e, int topk, int num_router_ctas, int w_hint, int pdl_mode, int epc, int k_parts) {
    stamp(stamps, 0);
    stamp_smid(stamps);
    if constexpr (kFullK) {
        if (pdl_mode == 1) pdl_trigger();
        uint32_t* cand = reinterpret_cast<uint32_t*>(logits);
        float* partials = reinterpret_cast<float*>(reinterpret_cast<char*>(logits) + kFullKPartialsOff);
        int* flags = reinterpret_cast<int*>(reinterpret_cast<char*>(logits) + kFullKFlagsOff);
        const int groups = num_router_ctas / k_parts;
        const int cand_slots = epc <= kCandSlots ? kCandSlots : kMaxCandSlots;
        if (static_cast<int>(blockIdx.x) >= num_router_ctas) {
            if constexpr (kCC > 0) {      // cc: the merger CTA's idle warps (8..) quantise the m rows while warps 0..7 poll
                if (w_hint & kTicketMergeBit) {      // ticket mode: this CTA only quantises (the last router CTA merges)
                    for (int t = 0; t < m; ++t) quant_cta<kMode, cc_block(kCC)>(hidden, x_bytes, x_sf, t, h);
                    stamp(stamps, 5);
                    if (pdl_mode == 2) pdl_trigger();
                    return;
                }
                if (threadIdx.x >= kThreads) {
                    constexpr int kQ = cc_block(kCC) - kThreads;
                    for (int t = 0; t < m; ++t) quant_role_range<kMode>(hidden, x_bytes, x_sf, t, h, threadIdx.x - kThreads, kQ, 1);
                    if (stamps != nullptr && threadIdx.x == kThreads) stamps[blockIdx.x * kStampSlots + 5] = globaltimer_ns();
                    return;
                }
            }
            const bool defer = (w_hint & kMergerDeferBit) != 0;
            const int nslots = groups * cand_slots;
            if (nslots <= 20 * 32)          // H20 full-K: 77 x 8 = 616 slots -> 20 per lane
                merger_role<kMaxTopK, 20>(cand, topk_idx, topk_weights, stamps, m, groups, cand_slots, defer);
            else if (nslots <= 32 * 32)
                merger_role<kMaxTopK, 32>(cand, topk_idx, topk_weights, stamps, m, groups, cand_slots, defer);
            else
                merger_role<kMaxTopK, kMaxSlotsPerLane>(cand, topk_idx, topk_weights, stamps, m, groups, cand_slots, defer);
        } else {
            if constexpr (kCC > 0) {
                uint32_t* ckeys = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(logits) + kCCKeysOff);
                const bool compact = (w_hint & kKeysCompactBit) != 0;
                router_role_fullk_cc<cc_slots(kCC), cc_ks(kCC), kW8>(hidden, router_weight, cand, ckeys, stamps, m, h, e, epc, groups, cand_slots, compact);
                if (w_hint & kSelectInMegaBit) {     // select in Mega: no ticket, no last arriver; kernel completion publishes the keys
                    stamp(stamps, 3);
                    if (pdl_mode == 2) pdl_trigger();
                    return;
                }
                if (w_hint & kTicketMergeBit) {
                    __shared__ int s_last;
                    const bool relaxed = (w_hint & kTicketRelaxedBit) != 0;
                    __syncthreads();                 // every key store of this CTA is issued before thread 0's ticket
                    if (threadIdx.x == 0) s_last = (relaxed ? atomicAdd(ticket, 1) : atom_add_acq_rel_gpu(ticket, 1)) == num_router_ctas - 1;
                    __syncthreads();
                    stamp(stamps, 3);
                    if (s_last) {
                        unsigned long long* mstamps = stamps != nullptr ? stamps + static_cast<size_t>(num_router_ctas) * kStampSlots : nullptr;
                        if (mstamps != nullptr && threadIdx.x == 0) mstamps[1] = globaltimer_ns();
                        if (compact)
                            last_arriver_topk_compact<kMaxTopK>(ckeys, topk_idx, topk_weights, mstamps, m, e, (w_hint & kTicketRelaxedBit) != 0, (w_hint & kSelectPrunedBit) != 0);
                        else if (groups * cand_slots <= 20 * 32)
                            last_arriver_topk<kMaxTopK, 20>(cand, topk_idx, topk_weights, mstamps, m, groups, cand_slots, (w_hint & kSelectReduxBit) != 0, (w_hint & kTicketRelaxedBit) != 0);
                        else
                            last_arriver_topk<kMaxTopK, kMaxSlotsPerLane>(cand, topk_idx, topk_weights, mstamps, m, groups, cand_slots, (w_hint & kSelectReduxBit) != 0, (w_hint & kTicketRelaxedBit) != 0);
                        if (threadIdx.x == 0) *ticket = 0;
                    }
                    if (pdl_mode == 2) pdl_trigger();
                    return;
                }
            } else if constexpr (kSwapBlk > 0) {
                router_role_fullk_swapab<kMode, kSwapBlk>(hidden, router_weight, cand, partials, flags, x_bytes, x_sf, stamps, m, h, e, epc, k_parts, groups, cand_slots, w_hint == 2 ? 1 : 0);
            } else if constexpr (kFma) {
                if (cand_slots == kCandSlots)
                    router_role_fullk_fma<kMode, kCandSlots, true>(hidden, router_weight, cand, partials, flags, x_bytes, x_sf, stamps, m, h, e, epc, k_parts, groups, cand_slots);
                else
                    router_role_fullk_fma<kMode, kMaxCandSlots, false>(hidden, router_weight, cand, partials, flags, x_bytes, x_sf, stamps, m, h, e, epc, k_parts, groups, cand_slots);
            } else {
                router_role_fullk<kMode>(hidden, router_weight, cand, partials, flags, x_bytes, x_sf, stamps, m, h, e, epc, k_parts, groups, cand_slots, w_hint & kMergerFlagsMask);
            }
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
        quant_cta<kMode>(hidden, x_bytes, x_sf, token, h);
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
                topk_tiny_cta<kKS, 12, kMaxTopK>(logits, topk_idx, topk_weights, stamps, token, m, e);
            else
                topk_tiny_cta<kKS, kMaxExperts / 32, kMaxTopK>(logits, topk_idx, topk_weights, stamps, token, m, e);
            // warps 1..7 returned inside topk_tiny_cta; warp 0 continues
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
    if constexpr (kSwapBlk > 0) router_role_swapab<kSwapBlk>(hidden, router_weight, logits, stamps, m, h, e, w_hint == 2 ? 1 : 0);
    else router_role<kMTiles, kTiny>(hidden, router_weight, logits, stamps, m, h, e, w_hint);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(ticket, 1);
    stamp(stamps, 3);
    if (pdl_mode == 2) pdl_trigger();
}

// ---------------------------------------------- lean cc kernel (DG_FE_CC_LEAN=1, round 5)
// Same numerics and hand-off as router_quant_topk_kernel<1, kMode, true, true, false, 0, 44> with the
// compact key array (e == 384, ticket merge), compiled as its own entry point. The generic cc44
// instantiation is 10.6 K SASS instructions and its router role starts 73 KB into the kernel (the
// entry branches over the polling-merger / slot-key / quant paths); after a 256 MB L2 flush (or the
// Mega weight stream) every CTA fetches that code cold from DRAM (NCU top stall no_instruction).
// Here the router CTA's hot path is the first code of the kernel and the cold paths (quant CTA,
// last-arriver select) are __noinline__ functions, and the router role issues fewer instructions
// after the weights land:
//   * weight chunks issued first (the DRAM-latency critical stream), then the L2-hot activations;
//   * activations converted to fp32 once, right after the loads (in the weight-latency shadow);
//   * bf16 -> fp32 as one LOP3 (high half) / one shift (low half) per element;
//   * row 1's conversion + FMA chain only when m == 2 (warp-uniform branch);
//   * the inactive slot of the last CTA (4 experts) loads row expert_base (valid memory) and
//     writes no key, instead of predicated loads.
// Accumulation order unchanged (per lane: fma chain over the 8 bf16 of chunk c in element order,
// c ascending; xor butterfly 16..1; K-part partials summed ks = 0..3; one bf16 rounding).
__device__ __forceinline__ float bf16lo_f32(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi_f32(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
template <int kMode>
__device__ __noinline__ void lean_quant_cta(const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
                                            float* __restrict__ x_sf, int m, int h) {
    for (int t = 0; t < m; ++t) quant_cta<kMode, cc_block(44)>(hidden, x_bytes, x_sf, t, h);
}
__device__ __noinline__ void lean_last_arriver(uint32_t* __restrict__ ckeys, int64_t* __restrict__ topk_idx,
                                               float* __restrict__ topk_weights, unsigned long long* mstamps,
                                               int m, int e, bool relaxed, bool pruned) {
    last_arriver_topk_compact<kMaxTopK>(ckeys, topk_idx, topk_weights, mstamps, m, e, relaxed, pruned);
}
template <int kMode>
__global__ void __launch_bounds__(cc_block(44), 1) router_cc_lean_kernel(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        int* __restrict__ ticket, float* __restrict__ logits,
        unsigned long long* stamps,
        int m, int h, int e, int topk, int num_router_ctas, int w_hint, int pdl_mode, int epc, int k_parts) {
    constexpr int kSlots = cc_slots(44), kKS = cc_ks(44);                 // 5 experts x 4 K-part warps
    constexpr int kKPart = kCCH / kKS, kChunks = kKPart / 8 / 32;         // 768 elements = 3 x 16 B chunks per lane
    __shared__ float part_s[kSlots][kCCMaxM][kKS];
    __shared__ int s_last;
    stamp(stamps, 0);
    stamp_smid(stamps);
    if (pdl_mode == 1) pdl_trigger();
    uint32_t* ckeys = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(logits) + kCCKeysOff);
    if (static_cast<int>(blockIdx.x) >= num_router_ctas) {                // spare CTA: quantise the m rows
        lean_quant_cta<kMode>(hidden, x_bytes, x_sf, m, h);
        stamp(stamps, 5);
        if (pdl_mode == 2) pdl_trigger();
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = warp / kKS, ks = warp % kKS;
    const int expert_base = blockIdx.x * epc;
    const int n_exp = min(epc, e - expert_base);
    const int wslot = slot < n_exp ? slot : 0;
    const int kbase = ks * kKPart + lane * 8;
    stamp(stamps, 7);
    const __nv_bfloat16* wr = router_weight + static_cast<int64_t>(expert_base + wslot) * h + kbase;
    uint4 wv[kChunks];
    #pragma unroll
    for (int c = 0; c < kChunks; ++c) wv[c] = ld_nc_na_16(wr + 256 * c);
    const bool two = m > 1;
    uint4 xv[kCCMaxM][kChunks];
    #pragma unroll
    for (int c = 0; c < kChunks; ++c) xv[0][c] = ld_nc_na_16(hidden + kbase + 256 * c);
    #pragma unroll
    for (int c = 0; c < kChunks; ++c) {
        xv[1][c] = make_uint4(0u, 0u, 0u, 0u);
        if (two) xv[1][c] = ld_nc_na_16(hidden + h + kbase + 256 * c);
    }
    stamp(stamps, 5);
    if (stamps != nullptr && threadIdx.x == 0) stamps[blockIdx.x * kStampSlots + 1] = globaltimer_after(wv[0].x);
    float xf[kCCMaxM][kChunks][8];
    #pragma unroll
    for (int c = 0; c < kChunks; ++c) {
        const uint32_t xw[4] = {xv[0][c].x, xv[0][c].y, xv[0][c].z, xv[0][c].w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) { xf[0][c][2 * j] = bf16lo_f32(xw[j]); xf[0][c][2 * j + 1] = bf16hi_f32(xw[j]); }
    }
    if (two) {
        #pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const uint32_t xw[4] = {xv[1][c].x, xv[1][c].y, xv[1][c].z, xv[1][c].w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) { xf[1][c][2 * j] = bf16lo_f32(xw[j]); xf[1][c][2 * j + 1] = bf16hi_f32(xw[j]); }
        }
    }
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (int c = 0; c < kChunks; ++c) {
        const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc0 = fmaf(bf16lo_f32(ww[j]), xf[0][c][2 * j], acc0);
            acc0 = fmaf(bf16hi_f32(ww[j]), xf[0][c][2 * j + 1], acc0);
        }
    }
    if (two) {
        #pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc1 = fmaf(bf16lo_f32(ww[j]), xf[1][c][2 * j], acc1);
                acc1 = fmaf(bf16hi_f32(ww[j]), xf[1][c][2 * j + 1], acc1);
            }
        }
    }
    acc0 = warp_sum(acc0);
    if (lane == 0) part_s[slot][0][ks] = acc0;
    if (two) {
        acc1 = warp_sum(acc1);
        if (lane == 0) part_s[slot][1][ks] = acc1;
    }
    __syncthreads();
    stamp(stamps, 2);
    if (static_cast<int>(threadIdx.x) < kSlots * kCCMaxM) {
        const int r = threadIdx.x / kSlots, c = threadIdx.x % kSlots;
        if (r < m && c < n_exp) {
            float v = 0.0f;
            #pragma unroll
            for (int k = 0; k < kKS; ++k) v += part_s[c][r][k];
            ckeys[static_cast<int64_t>(r) * e + expert_base + c] = topk_key(round_bf16(v), expert_base + c);
        }
    }
    if (w_hint & kSelectInMegaBit) {          // select in Mega: kernel completion publishes the keys
        stamp(stamps, 3);
        if (pdl_mode == 2) pdl_trigger();
        return;
    }
    const bool relaxed = (w_hint & kTicketRelaxedBit) != 0;
    __syncthreads();                          // every key store of this CTA is issued before thread 0's ticket
    if (threadIdx.x == 0) s_last = (relaxed ? atomicAdd(ticket, 1) : atom_add_acq_rel_gpu(ticket, 1)) == num_router_ctas - 1;
    __syncthreads();
    stamp(stamps, 3);
    if (s_last) {
        unsigned long long* mstamps = stamps != nullptr ? stamps + static_cast<size_t>(num_router_ctas) * kStampSlots : nullptr;
        if (mstamps != nullptr && threadIdx.x == 0) mstamps[1] = globaltimer_ns();
        lean_last_arriver(ckeys, topk_idx, topk_weights, mstamps, m, e, relaxed, (w_hint & kSelectPrunedBit) != 0);
        if (threadIdx.x == 0) *ticket = 0;
    }
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
// one merger CTA; groups = floor((T - 1) / k_parts) expert groups of epc = ceil(e /
// groups) experts (<= 16), re-packed to groups = ceil(e / epc); router CTAs = groups x
// k_parts (H20, k_parts 1: 77 x 5; k_parts 2: 35 groups x 11 x 2 = 70; T = 97,
// k_parts 4: 24 x 16 x 4 = 96).
static void fullk_plan(int e, int grid, int k_parts, int& epc, int& router_ctas) {
    const int target = (grid <= 0 ? num_sms_cached() : grid) - 1;
    const int groups0 = std::max(target / k_parts, 1);
    epc = (e + groups0 - 1) / groups0;
    epc = std::min(std::max(epc, 1), kMaxCandSlots);
    const int groups = (e + epc - 1) / epc;
    router_ctas = groups * k_parts;
}

template <int kMTiles, int kMode, bool kTiny, bool kFullK, bool kFma = false, int kSwapBlk = 0, int kCC = 0, bool kW8 = false>
void launch(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
            int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
            int m, int h, int e, int topk, int l2_persist, int pdl_mode, int grid, int k_parts, cudaStream_t stream, int mma = 0) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    int epc = kExpertsPerCTA, router_ctas = ((e + kExpertsPerCTA - 1) / kExpertsPerCTA) * Cfg::kKSplitCTAs;
    int smem_bytes = Cfg::kDynSmemBytes, num_ctas = router_ctas + m;
    if constexpr (kFullK) {
        fullk_plan(e, grid, k_parts, epc, router_ctas);
        smem_bytes = kCC > 0 ? 0 : kSwapBlk > 0 ? swapab_smem_bytes(m, h / k_parts)
                  : kFma ? (kThreads / 32) * kMaxCandSlots * 16 * 4 : fullk_smem_bytes(m, h / k_parts);
        num_ctas = router_ctas + 1;
        // DG_FE_FULLK_1PERSM (default 1): when the grid fits the SM count, request
        // enough dynamic smem that only one CTA fits per SM, so the block scheduler
        // cannot pack two router CTAs (and their in-flight loads) onto one SM.
        static const int one_per_sm = getenv("DG_FE_FULLK_1PERSM") ? atoi(getenv("DG_FE_FULLK_1PERSM")) : 1;
        if (one_per_sm && num_ctas <= num_sms_cached()) smem_bytes = std::max(smem_bytes, 116 * 1024);
    }
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(router_quant_topk_kernel<kMTiles, kMode, kTiny, kFullK, kFma, kSwapBlk, kCC, kW8>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kFullK ? std::max(fullk_smem_bytes(16, 3072), 116 * 1024) : Cfg::kDynSmemBytes);
        attr_set = true;
    }
    const size_t w_bytes = kW8 ? static_cast<size_t>(e) * (h + kCCW8Pad) : static_cast<size_t>(e) * h * sizeof(__nv_bfloat16);
    // cc default: router weights in the persisting L2 set-aside (measured -0.25..-0.5 us kernel end on H20);
    // DG_FE_ROUTER_L2_PERSIST=0 opts out explicitly.
    if (kCC > 0 && l2_persist == 0 && getenv("DG_FE_ROUTER_L2_PERSIST") == nullptr) l2_persist = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(num_ctas);
    cfg.blockDim = dim3(cc_block(kCC));
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
    // swapab paths reuse w_hint as the weight-layout flag: 2 = fragment layout (mma == 3);
    // the L2::evict_last hint (l2_persist == 2) is not implemented for swapab.
    int w_hint = kSwapBlk > 0 ? (mma == 3 ? 2 : 0) : (l2_persist == 2 ? 1 : 0);
    if constexpr (kFullK) {
        static const int defer = getenv("DG_FE_MERGER_DEFER") ? atoi(getenv("DG_FE_MERGER_DEFER")) : (kCC > 0 ? 1 : 0);
        if (defer) w_hint |= kMergerDeferBit;
        // cc default: ticket (last-arriver merge). H20 kernel end rows 1|2 x mxfp4|qoq: polling merger 5.12-5.63 us,
        // ticket 4.86-5.12 (fence + atomic 0.3-0.5 after the last keys, read 0.25, fold + merge 0.5, softmax + write 0.25).
        static const bool ticket_merge = getenv("DG_FE_CC_MERGE") ? strcmp(getenv("DG_FE_CC_MERGE"), "ticket") == 0 : true;
        if (kCC > 0 && ticket_merge) w_hint |= kTicketMergeBit;
        static const bool redux_sel = getenv("DG_FE_CC_SELECT") && strcmp(getenv("DG_FE_CC_SELECT"), "redux") == 0;
        if (kCC > 0 && redux_sel) w_hint |= kSelectReduxBit;
        // ticket: relaxed (default) = atomicAdd right after the key stores, the last CTA re-reads slots still 0;
        // acqrel = atom.acq_rel.gpu (release drains this CTA's stores first: ~0.5 us more before the ticket)
        static const bool acqrel = getenv("DG_FE_CC_TICKET") && strcmp(getenv("DG_FE_CC_TICKET"), "acqrel") == 0;
        if (kCC > 0 && !acqrel) w_hint |= kTicketRelaxedBit;
        // keys: compact [token][e] array (default when e == 384: 12 keys per lane for the last arriver) | slots
        static const bool slots = getenv("DG_FE_CC_KEYS") && strcmp(getenv("DG_FE_CC_KEYS"), "slots") == 0;
        if (kCC > 0 && !slots && e == 384) w_hint |= kKeysCompactBit;
        if (kCC > 0 && mma == 8) w_hint |= kSelectInMegaBit | kKeysCompactBit | kTicketMergeBit;   // spare CTA only quantises
        // round 5: DG_FE_CC_SELECT=pruned (two-level threshold select, compact keys only; same result)
        static const bool pruned_sel = getenv("DG_FE_CC_SELECT") && strcmp(getenv("DG_FE_CC_SELECT"), "pruned") == 0;
        if (kCC > 0 && pruned_sel && (w_hint & kKeysCompactBit)) w_hint |= kSelectPrunedBit;
    }
    if constexpr (kCC == 44 && !kW8) {
        // round 5: DG_FE_CC_LEAN (default 1) -> dedicated cc44 entry point (hot path first, cold paths out of line, leaner
        // FMA body); 0 = the generic router_quant_topk_kernel<..., 44> instantiation
        static const bool lean = getenv("DG_FE_CC_LEAN") ? atoi(getenv("DG_FE_CC_LEAN")) != 0 : true;
        if (lean && (w_hint & kKeysCompactBit) && (w_hint & kTicketMergeBit)) {
            static bool lean_attr = false;
            if (!lean_attr) {
                cudaFuncSetAttribute(router_cc_lean_kernel<kMode>, cudaFuncAttributeMaxDynamicSharedMemorySize, 116 * 1024);
                lean_attr = true;
            }
            cudaLaunchKernelEx(&cfg, router_cc_lean_kernel<kMode>,
                               hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, router_ctas, w_hint, pdl_mode, epc, k_parts);
            return;
        }
    }
    cudaLaunchKernelEx(&cfg, router_quant_topk_kernel<kMTiles, kMode, kTiny, kFullK, kFma, kSwapBlk, kCC, kW8>,
                       hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, router_ctas, w_hint, pdl_mode, epc, k_parts);
}

template <int kMode>
void launch_mode(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
                 int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
                 int m, int h, int e, int topk, bool tiny, int l2_persist, int pdl_mode, int grid, int mma, int k_parts, cudaStream_t stream) {
    k_parts = (k_parts == 2 || k_parts == 4) ? k_parts : 1;
    const int kpart_len = h / k_parts;
    if (m <= 16 && tiny && topk == kMaxTopK && grid != 96 && kpart_len % kChunkK == 0 && kpart_len <= 3072 && h % k_parts == 0) {
        int epc = 0, rc = 0;
        fullk_plan(e, grid, k_parts, epc, rc);
        // fma: <= 8 experts -> 2 vectors/thread (kpart_len <= 4096); > 8 experts -> 1 vector/thread (kpart_len <= 2048)
        const bool fma_ok = mma == 1 && (epc <= kCandSlots ? kpart_len <= 8 * 2 * kThreads : kpart_len <= 8 * kThreads);
        if (rc <= kMaxRouterCTAs && (rc / k_parts) * (epc <= kCandSlots ? kCandSlots : kMaxCandSlots) <= kMaxSlotsPerLane * 32) {
            // swapab: kBlk = K-part / 256 blocks per warp (h = 3072: 12 | 6 | 3); other shapes fall back to WMMA
            const int blk = kpart_len / kChunkK;
            if (mma == 3 && epc != kExpertsPerCTA) {
                static bool warned = false;
                if (!warned) { fprintf(stderr, "[fable_frontend] DG_FE_ROUTER_WLAYOUT=fragment needs 16-expert groups (epc=%d): using row layout\n", epc); warned = true; }
                mma = 2;
            }
            // cc | cc6: CUDA-core K-split router (m <= 2, h = 3072, k_parts 1, epc <= 5); else WMMA
            if (mma == 4 || mma == 5 || mma == 6 || mma == 7 || mma == 8) {
                const int cc = (mma == 4 || mma == 8) ? 44 : mma == 5 ? 46 : mma == 6 ? 36 : 43;      // slots * 8 + K-split (ccfp8: 5 x 3); 8 = cc + select in Mega
                if (m <= kCCMaxM && h == kCCH && k_parts == 1 && epc <= cc_slots(cc)) {
                    if (cc == 43) launch<1, kMode, true, true, false, 0, 43, true>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream);
                    else if (cc == 44) launch<1, kMode, true, true, false, 0, 44>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream, mma);
                    else if (cc == 46) launch<1, kMode, true, true, false, 0, 46>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream);
                    else launch<1, kMode, true, true, false, 0, 36>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream);
                    return;
                }
                static bool warned_cc = false;
                if (!warned_cc) { fprintf(stderr, "[fable_frontend] DG_FE_TINYM_MMA=cc needs m <= %d, h = %d, k_parts 1, epc <= %d (m=%d h=%d k_parts=%d epc=%d): using WMMA\n", kCCMaxM, kCCH, cc_slots(cc), m, h, k_parts, epc); warned_cc = true; }
                mma = 0;
            }
            const bool swap = mma == 2 || mma == 3;
            if (swap && blk == 12) launch<1, kMode, true, true, false, 12>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream, mma);
            else if (swap && blk == 6) launch<1, kMode, true, true, false, 6>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream, mma);
            else if (swap && blk == 3) launch<1, kMode, true, true, false, 3>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream, mma);
            else if (fma_ok) launch<1, kMode, true, true, true>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream);
            else launch<1, kMode, true, true, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, k_parts, stream);
            return;
        }
    }
    // legacy 96 x 4 grid with the swapped MMA (h / 256 / 4 == 3 blocks per warp, i.e. h == 3072)
    if (m <= 16 && tiny && topk == kMaxTopK && (mma == 2 || mma == 3) && h == 3 * kChunkK * RouterCfg<1, true>::kKSplitCTAs)
        launch<1, kMode, true, false, false, 3>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, 1, stream, mma);
    else if (m <= 16 && tiny && topk == kMaxTopK) launch<1, kMode, true, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, 1, stream);
    else if (m <= 16) launch<1, kMode, false, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, 1, stream);
    else if (m <= 32) launch<2, kMode, false, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, 1, stream);
    else launch<4, kMode, false, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, grid, 1, stream);
}

}  // namespace

size_t router_quant_topk_frontend_workspace_bytes(int e) {
    return kFrontendStampsOffsetBase + static_cast<size_t>(4) * 64 * e * 4 + kFrontendStampsBytes;
}

int router_quant_topk_frontend_router_ctas(int m, int h, int e, int topk, int tiny, int grid, int k_parts) {
    k_parts = (k_parts == 2 || k_parts == 4) ? k_parts : 1;
    if (m <= 16 && tiny && topk == kMaxTopK && grid != 96 && (h / k_parts) % kChunkK == 0 && h / k_parts <= 3072) {
        int epc = 0, router_ctas = 0;
        fullk_plan(e, grid, k_parts, epc, router_ctas);
        if (router_ctas <= kMaxRouterCTAs) return router_ctas;
    }
    const int groups = (e + kExpertsPerCTA - 1) / kExpertsPerCTA;
    return groups * (m <= 16 ? 4 : 2);
}

void launch_router_quant_topk_frontend(
        const void* hidden, const void* router_weight,
        void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
        void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
        int tiny, int stamps_on, int l2_persist, int pdl_mode, int grid, int mma, int k_parts, cudaStream_t stream) {
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
                       ticket, logits, stamps, m, h, e, topk, use_tiny, l2_persist, pdl_mode, grid, mma, k_parts, stream);
    else
        launch_mode<1>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, stamps, m, h, e, topk, use_tiny, l2_persist, pdl_mode, grid, mma, k_parts, stream);
}
