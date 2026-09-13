// Fused router + top-k + activation-quantization frontend for the SM90 fused
// MegaMoE (see fable_frontend.h). Three launch shapes, chosen by `select_fe_path`:
//   * cc router (m <= 2 rows, h = 3072, e = 384, top-8; the customer's M = 2..16 on 8
//     ranks): `router_cc_lean_kernel`, 77 CTAs x 5 experts x 4 K-part warps stream the
//     router weights straight into registers (CUDA-core FMA), write one 32-bit key per
//     (token, expert) into a compact [token][e] array, and either the last-arriving CTA
//     (atomic ticket) selects the top-8 + softmax, or (select_in_mega) the fused MegaMoE
//     prologue does (deep_gemm/impls/fable_cc_select.cuh). The spare CTA quantises.
//   * swapab (m <= 16, h = 3072, top-8): legacy 96-CTA grid (24 expert groups x 4 K-parts)
//     with the experts on the MMA M dimension (mma.sync m16n8k16 bf16), weight A fragments
//     ld.global.nc straight into registers (fragment-permuted weights, see the header),
//     plus m quant / top-k CTAs.
//   * wmma (every other shape): 96-CTA (m <= 16) or 48-CTA (m <= 64) grid, cp.async smem
//     ring + WMMA bf16 m16n16k16 fp32 accumulate, plus m quant / top-k CTAs.
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
// Device code (router WMMA, tiny top-k) lives in deep_gemm/include/deep_gemm/impls/
// fable_frontend_device.cuh; this file keeps the launch, the CTA roles, the quantisation,
// the swapab router and the cc router.
using namespace deep_gemm::fable_fe;

struct CtaSync {
    __device__ __forceinline__ void operator()() const { __syncthreads(); }
};

// Router CTA: unit == blockIdx.x (16 experts x one K-part), dynamic smem.
template <int kMTiles, bool kTiny>
__device__ __forceinline__ void router_role(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        float* __restrict__ logits,          // [k_part][m][e] workspace partials
        unsigned long long* stamps,
        int m, int h, int e) {
    extern __shared__ __align__(128) uint8_t dyn_smem[];
    router_unit<kMTiles, kTiny>(hidden, router_weight, logits, stamps, m, h, e,
                                static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x), dyn_smem, CtaSync{});
}

// Top-k (largest value, smallest expert id on ties) + softmax over the selected
// bf16 logits. One warp per token, lane owns experts lane, lane+32, ...
__device__ __forceinline__ void topk_softmax_token(
        const float* __restrict__ logits, int64_t* __restrict__ topk_idx,
        float* __restrict__ topk_weights, int t, int m, int e, int topk, int k_split) {
    const int lane = threadIdx.x & 31;
    constexpr int kPerLane = kMaxExperts / 32;
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

// Tiny-M top-k CTA wrapper (static smem for the candidate keys).
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
// thread subset). smem_slot: which 32-entry slice of the warp-max scratch this thread group uses
// (two groups quantising two rows concurrently in one CTA, cc kernel, use slots 0 and 1 with their
// own named barriers).
template <int kMode>
__device__ __forceinline__ void quant_role_range(
        const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
        float* __restrict__ x_sf, int token, int h, int tid, int nthreads, int bar_id, int smem_slot = 0) {
    __shared__ float smem_warp_max_all[2][32];
    float* smem_warp_max = smem_warp_max_all[smem_slot];
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

// ------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t smid_u32() {
    uint32_t v;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
    return v;
}
// stamp slot 6 of every CTA = %smid (placement / packing diagnostics, not a time)
__device__ __forceinline__ void stamp_smid(unsigned long long* stamps) {
    if (stamps != nullptr && threadIdx.x == 0) stamps[blockIdx.x * kStampSlots + 6] = smid_u32();
}
__device__ __forceinline__ uint4 ld_nc_na_16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
// globaltimer read predicated on `dep`: cannot be scheduled before the load producing it lands
__device__ __forceinline__ unsigned long long globaltimer_after(uint32_t dep) {
    unsigned long long t;
    asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %1, 0x7fffffff;\n @p mov.u64 %0, %%globaltimer;\n @!p mov.u64 %0, 0;\n}"
                 : "=l"(t) : "r"(dep));
    return t;
}
using fable_cc::merge8;
using fable_cc::topk_finish;
using fable_cc::ld_cg_v4;

// ------------------------------------------------------- swapped-operand MMA
// The router product is computed as
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
// K ownership / accumulation order (fixed, documented): warp w owns K-slice
// [32w, 32w + 32) of every 256-wide chunk of its K-part, block j = chunk j; the
// warp accumulates its blocks in order j = 0..kBlk-1 (steps s = 0, 1 each) in ONE
// fp32 accumulator (the 16-product sum inside one m16n8k16 is the hardware order);
// the 8 warp partials are summed in warp order 0..7 through smem; then the
// 4 K-part partials in order ks = 0..3 in the top-k CTA; one bf16 rounding of the
// logit. Deterministic, not bit-identical to the WMMA path.
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
    // part_s[warp][token * 16 + expert] (the same [row][col] layout the WMMA path stores).
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

// ------------------------------------------------- legacy grid kernel (wmma | swapab)
// Router CTAs (E / 16 groups x K-parts) then m quant / top-k CTAs; the top-k CTAs wait for
// the router ticket. kTiny (m <= 16): <= 85 regs/thread so 3 CTAs (59 KB smem each) fit
// per SM -> single wave on 78 SMs. kSwapBlk > 0: swapab router (kBlk = K-part / 256).
template <int kMTiles, int kMode, bool kTiny, int kSwapBlk>
__global__ void __launch_bounds__(kThreads, kTiny ? 3 : 1) router_quant_topk_kernel(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        int* __restrict__ ticket, float* __restrict__ logits,
        unsigned long long* stamps,
        int m, int h, int e, int topk, int num_router_ctas, int wlayout) {
    stamp(stamps, 0);
    stamp_smid(stamps);
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
    if constexpr (kSwapBlk > 0) router_role_swapab<kSwapBlk>(hidden, router_weight, logits, stamps, m, h, e, wlayout);
    else router_role<kMTiles, kTiny>(hidden, router_weight, logits, stamps, m, h, e);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(ticket, 1);
    stamp(stamps, 3);
}

// ---------------------------------------------- CUDA-core K-split router (cc)
// m <= 2, h = 3072, e = 384, top-8: CTA of 5 experts x 4 K-part warps (640 threads). Warp
// (slot s, ks) owns expert expert_base + s over K-part ks (768 elements); lane l owns the
// 16 B chunks l, l + 32, l + 64 of that part. The kernel's first instructions issue every
// weight chunk of the CTA straight into registers (ld.global.nc.L1::no_allocate.v4; no smem,
// no TMA, no barrier before the FMAs), then the L2-hot activation chunks; activations are
// converted to fp32 once in the weight-latency shadow (bf16 -> fp32 = one LOP3 / one shift),
// row 1 only when m == 2. Accumulation order (fixed): per lane, fma chain over the 8 bf16
// of chunk c in element order, c ascending; xor butterfly (16, 8, 4, 2, 1) over the lanes;
// K-part partials summed ks = 0..3 through smem; one bf16 rounding. Every (token, expert)
// logit becomes a 32-bit key (value desc, id asc) in the compact [token][e] array at
// workspace + 256 + kCCKeysOff. Hand-off: keys stored -> bar.sync -> thread 0 relaxed atomic
// ticket; the LAST router CTA reads the 384 keys per token (3 x 16 B per lane, re-reading
// slots still 0), per-lane sorted top-8 + one 8-round redux merge, softmax, writes top-k.
// select_in_mega: no ticket / no select -- the router CTAs end after their keys and the
// fused MegaMoE prologue selects (deep_gemm/impls/fable_cc_select.cuh). zero_row_unrouted:
// an all-zero hidden row gets unrouted keys instead of logits (see kCCZeroRowKey below). The
// spare CTA (blockIdx == num_router_ctas) quantises the m rows (both concurrently when m == 2).
// The hot path is the first code of the kernel; the cold paths (quant CTA, last-arriver
// select) are __noinline__ (2.7 K SASS instructions: after a 256 MB L2 flush or the Mega
// weight stream every CTA fetches its code cold from DRAM).
constexpr int kCCMaxM = 2;
constexpr int kCCH = 3072;
constexpr int kCCE = 384;
constexpr int kCCSlots = 5, kCCKS = 4;                              // 5 experts x 4 K-part warps
constexpr int kCCBlock = kCCSlots * kCCKS * 32;                     // 640 threads
constexpr size_t kCCKeysOff = 64 * 1024;                            // workspace offset (after the 256 B ticket area) of the compact keys ([2][512] u32 max)
// All-zero hidden row (zero_row_unrouted, DG_FE_ZERO_ROW_UNROUTED): x = 0 -> every expert output is 0 -> y = 0
// exactly, so the row is UNROUTED instead of tie-broken onto experts 0..7 (all on rank 0, weight 1/8). The
// router CTAs OR-reduce the row they already hold in registers (sign bit masked: -0.0 counts as zero) and
// write, for such a row, every one of its 384 keys as
//   select_in_mega:  0u              -- the fused Mega prologue's "unrouted token" encoding (topk_idx -1 / weight 0)
//   FE select:       kCCZeroRowKey   -- non-zero (the last-arriver spin treats 0 as "not yet visible"); its low
//                                       16 bits = 0xFFFF - expert would mean expert 0xFFFE, which no real key has
// and the last arriver writes topk_idx -1 / topk_weights 0 for it. Non-zero rows: unchanged bits.
constexpr uint32_t kCCZeroRowKey = 1u;

// Compact-key last-arriver select: token t's e (= 384) keys are contiguous -> 3 x 16 B per lane,
// insertion 12 x 8 + merge8, softmax, top-k write, keys zeroed for the next launch.

// Compact-key last-arriver select: token t's e (= 384) keys are contiguous -> 3 x 16 B per lane,
// insertion 12 x 8 + merge8, softmax, top-k write, keys zeroed for the next launch. A row whose
// keys are kCCZeroRowKey (all-zero hidden row, see above) is written unrouted (-1 / 0).
template <int kTopK>
__device__ __forceinline__ void last_arriver_topk_compact(
        uint32_t* __restrict__ ckeys, int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        unsigned long long* mstamps, int m, int e) {
    constexpr int kVec = 3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= m) return;
    const int t = warp;
    uint32_t* base = ckeys + static_cast<int64_t>(t) * e;
    uint4 q[kVec];
    #pragma unroll
    for (int i = 0; i < kVec; ++i) q[i] = ld_cg_v4(base + 4 * (lane + 32 * i));
    if (mstamps != nullptr && t == 0 && lane == 0) mstamps[2] = globaltimer_after(q[0].x);
    // relaxed ticket: a slot may not be visible yet -> the data is the flag, re-read zeros
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
    // all-zero hidden row (every key of the row == kCCZeroRowKey, key 0 suffices): unrouted
    if (__shfl_sync(0xffffffffu, q[0].x, 0) == kCCZeroRowKey) {
        if (lane < kTopK) {
            topk_idx[static_cast<int64_t>(t) * kTopK + lane] = -1;
            topk_weights[static_cast<int64_t>(t) * kTopK + lane] = 0.0f;
        }
        if (mstamps != nullptr && t == 0 && lane == 0) mstamps[3] = mstamps[4] = globaltimer_ns();
    } else {
        uint32_t run = 0u;
        uint32_t loc[kTopK];
        fable_cc::insert_keys<kTopK, kVec>(q, loc);
        merge8<kTopK>(run, loc, lane);
        if (mstamps != nullptr && t == 0 && lane == 0) mstamps[3] = globaltimer_ns();
        topk_finish<kTopK>(run, lane, t, topk_idx, topk_weights);
        if (mstamps != nullptr && t == 0 && lane == 0) mstamps[4] = globaltimer_ns();
    }
    #pragma unroll
    for (int i = 0; i < kVec; ++i) *reinterpret_cast<uint4*>(base + 4 * (lane + 32 * i)) = make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ float bf16lo_f32(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi_f32(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
// Spare CTA: m == 1 -> the whole CTA quantises the row; m == 2 -> the two rows concurrently, 320 threads (10 warps)
// each with their own named barrier (1, 2) and warp-max scratch slot (the row amax is order independent, per-element
// rounding unchanged, mxfp4 K128 groups still map to aligned half-warps).
template <int kMode>
__device__ __noinline__ void lean_quant_cta(const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
                                            float* __restrict__ x_sf, int m, int h) {
    if (m == 1) { quant_cta<kMode, kCCBlock>(hidden, x_bytes, x_sf, 0, h); return; }
    constexpr int kHalf = kCCBlock / 2;                              // 320
    const int r = threadIdx.x / kHalf, tid = threadIdx.x % kHalf;
    if (r < m) quant_role_range<kMode>(hidden, x_bytes, x_sf, r, h, tid, kHalf, 1 + r, r);
}
__device__ __noinline__ void lean_last_arriver(uint32_t* __restrict__ ckeys, int64_t* __restrict__ topk_idx,
                                               float* __restrict__ topk_weights, unsigned long long* mstamps,
                                               int m, int e) {
    last_arriver_topk_compact<kMaxTopK>(ckeys, topk_idx, topk_weights, mstamps, m, e);
}
// Stamp slots ([blockIdx][8] u64 ns; DG_FE_STAMPS=1): router CTA 0 start / 1 chunk0 landed /
// 2 logits done / 3 keys written (+ ticket) / 5 loads issued / 6 %smid / 7 prologue done;
// spare CTA 0 start / 5 quant done / 6 %smid; last arriver (in the spare CTA's slots) 1 ticket
// won / 2 keys read / 3 select done / 4 top-k written.
template <int kMode>
__global__ void __launch_bounds__(kCCBlock, 1) router_cc_lean_kernel(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        int* __restrict__ ticket, float* __restrict__ logits,
        unsigned long long* stamps,
        int m, int h, int e, int num_router_ctas, int epc, int select_in_mega, int zero_row_unrouted) {
    constexpr int kSlots = kCCSlots, kKS = kCCKS;
    constexpr int kKPart = kCCH / kKS, kChunks = kKPart / 8 / 32;         // 768 elements = 3 x 16 B chunks per lane
    __shared__ float part_s[kSlots][kCCMaxM][kKS];
    __shared__ uint32_t nz_s[kSlots][kCCMaxM][kKS];                       // per-warp OR of the row's bf16 bits (sign masked)
    __shared__ int s_last;
    stamp(stamps, 0);
    stamp_smid(stamps);
    uint32_t* ckeys = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(logits) + kCCKeysOff);
    if (static_cast<int>(blockIdx.x) >= num_router_ctas) {                // spare CTA: quantise the m rows
        lean_quant_cta<kMode>(hidden, x_bytes, x_sf, m, h);
        stamp(stamps, 5);
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = warp / kKS, ks = warp % kKS;
    const int expert_base = blockIdx.x * epc;
    const int n_exp = min(epc, e - expert_base);
    const int wslot = slot < n_exp ? slot : 0;              // inactive slot of the last CTA: valid memory, no key
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
    if (zero_row_unrouted) {
        // row-is-zero flag: OR of the row's bf16 bits (the 4 K-part warps of a slot hold the whole row), sign masked
        uint32_t nz0 = 0u, nz1 = 0u;
        #pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            nz0 |= xv[0][c].x | xv[0][c].y | xv[0][c].z | xv[0][c].w;
            nz1 |= xv[1][c].x | xv[1][c].y | xv[1][c].z | xv[1][c].w;
        }
        nz0 = __reduce_or_sync(0xffffffffu, nz0 & 0x7FFF7FFFu);
        nz1 = __reduce_or_sync(0xffffffffu, nz1 & 0x7FFF7FFFu);
        if (lane == 0) { nz_s[slot][0][ks] = nz0; nz_s[slot][1][ks] = nz1; }
    }
    __syncthreads();
    stamp(stamps, 2);
    if (static_cast<int>(threadIdx.x) < kSlots * kCCMaxM) {
        const int r = threadIdx.x / kSlots, c = threadIdx.x % kSlots;
        if (r < m && c < n_exp) {
            float v = 0.0f;
            uint32_t nz = 1u;
            #pragma unroll
            for (int k = 0; k < kKS; ++k) v += part_s[c][r][k];
            if (zero_row_unrouted) {
                nz = 0u;
                #pragma unroll
                for (int k = 0; k < kKS; ++k) nz |= nz_s[c][r][k];
            }
            ckeys[static_cast<int64_t>(r) * e + expert_base + c] = nz != 0u ? topk_key(round_bf16(v), expert_base + c)
                                                                            : (select_in_mega ? 0u : kCCZeroRowKey);
        }
    }
    if (select_in_mega) {                     // kernel completion publishes the keys; the Mega prologue selects
        stamp(stamps, 3);
        return;
    }
    __syncthreads();                          // every key store of this CTA is issued before thread 0's ticket
    if (threadIdx.x == 0) s_last = atomicAdd(ticket, 1) == num_router_ctas - 1;
    __syncthreads();
    stamp(stamps, 3);
    if (s_last) {
        unsigned long long* mstamps = stamps != nullptr ? stamps + static_cast<size_t>(num_router_ctas) * kStampSlots : nullptr;
        if (mstamps != nullptr && threadIdx.x == 0) mstamps[1] = globaltimer_ns();
        lean_last_arriver(ckeys, topk_idx, topk_weights, mstamps, m, e);
        if (threadIdx.x == 0) *ticket = 0;
    }
}

// DG_FE_ROUTER_L2_PERSIST=1 (the cc router's default): CUDA persisting-L2 set-aside for the
// router weights. Once per process: carve `bytes` (clamped to
// cudaDevAttrMaxPersistingL2CacheSize) out of L2; per launch: an access-policy-window launch
// attribute over the router weight buffer (hitProp Persisting). Launch attributes are recorded
// on the kernel node under stream capture, so eager and CUDA-graph paths behave the same.
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
// cc plan: one router CTA per SM minus the spare CTA; epc = ceil(e / (SMs - 1)) experts per CTA
// (<= kCCSlots), router CTAs = ceil(e / epc) (H20: 77 x 5 for e = 384).
static void cc_plan(int e, int& epc, int& router_ctas) {
    const int target = std::max(num_sms_cached() - 1, 1);
    epc = std::min(std::max((e + target - 1) / target, 1), kCCSlots);
    router_ctas = (e + epc - 1) / epc;
}

// Launch-config helper: optional persisting-L2 access-policy window over the router weights.
static void make_launch_config(cudaLaunchConfig_t& cfg, cudaLaunchAttribute* attrs, int num_ctas, int block,
                               int smem_bytes, cudaStream_t stream, const void* w, size_t w_bytes, int l2_persist) {
    cfg = {};
    cfg.gridDim = dim3(num_ctas);
    cfg.blockDim = dim3(block);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 0;
    if (l2_persist == 1) {
        set_persisting_l2_once(w_bytes);
        int max_window = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev);
        auto& a = attrs[cfg.numAttrs++];
        a.id = cudaLaunchAttributeAccessPolicyWindow;
        a.val.accessPolicyWindow.base_ptr = const_cast<void*>(w);
        a.val.accessPolicyWindow.num_bytes = std::min(w_bytes, static_cast<size_t>(max_window));
        a.val.accessPolicyWindow.hitRatio = 1.0f;
        a.val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        a.val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    }
}

template <int kMTiles, int kMode, bool kTiny, int kSwapBlk = 0>
void launch_legacy(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
                   int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
                   int m, int h, int e, int topk, int l2_persist, int wlayout, cudaStream_t stream) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    const int router_ctas = ((e + kExpertsPerCTA - 1) / kExpertsPerCTA) * Cfg::kKSplitCTAs;
    const int smem_bytes = kSwapBlk > 0 ? swapab_smem_bytes(m, h / Cfg::kKSplitCTAs) : Cfg::kDynSmemBytes;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(router_quant_topk_kernel<kMTiles, kMode, kTiny, kSwapBlk>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kDynSmemBytes);
        attr_set = true;
    }
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attrs[1];
    make_launch_config(cfg, attrs, router_ctas + m, kThreads, smem_bytes, stream, w,
                       static_cast<size_t>(e) * h * sizeof(__nv_bfloat16), l2_persist);
    cudaLaunchKernelEx(&cfg, router_quant_topk_kernel<kMTiles, kMode, kTiny, kSwapBlk>,
                       hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, router_ctas, kSwapBlk > 0 ? wlayout : 0);
}

template <int kMode>
void launch_cc(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
               int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
               int m, int h, int e, int l2_persist, int select_in_mega, int zero_row_unrouted, cudaStream_t stream) {
    int epc = 0, router_ctas = 0;
    cc_plan(e, epc, router_ctas);
    const int num_ctas = router_ctas + 1;
    // When the grid fits the SM count, request enough dynamic smem that only one CTA fits per
    // SM, so the block scheduler cannot pack two router CTAs (and their in-flight loads) onto
    // one SM.
    const int smem_bytes = num_ctas <= num_sms_cached() ? 116 * 1024 : 0;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(router_cc_lean_kernel<kMode>, cudaFuncAttributeMaxDynamicSharedMemorySize, 116 * 1024);
        attr_set = true;
    }
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attrs[1];
    make_launch_config(cfg, attrs, num_ctas, kCCBlock, smem_bytes, stream, w,
                       static_cast<size_t>(e) * h * sizeof(__nv_bfloat16), l2_persist);
    cudaLaunchKernelEx(&cfg, router_cc_lean_kernel<kMode>,
                       hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, router_ctas, epc, select_in_mega, zero_row_unrouted);
}

template <int kMode>
void launch_mode(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
                 int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
                 int m, int h, int e, int topk, int l2_persist, int wlayout, int select_in_mega, int zero_row_unrouted, cudaStream_t stream) {
    switch (select_fe_path(m, h, e, topk)) {
        case kFEPathCC:
            launch_cc<kMode>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, l2_persist, select_in_mega, zero_row_unrouted, stream);
            return;
        case kFEPathSwapAB:   // legacy 96 x 4 grid, h / 256 / 4 == 3 blocks per warp (h == 3072)
            launch_legacy<1, kMode, true, 3>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, wlayout, stream);
            return;
        default:
            break;
    }
    if (m <= 16 && topk == kMaxTopK) launch_legacy<1, kMode, true>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, 0, stream);
    else if (m <= 16) launch_legacy<1, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, 0, stream);
    else if (m <= 32) launch_legacy<2, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, 0, stream);
    else launch_legacy<4, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, 0, stream);
}

}  // namespace

int select_fe_path(int m, int h, int e, int topk) {
    if (m <= kCCMaxM && h == kCCH && e == kCCE && topk == kMaxTopK) return kFEPathCC;
    if (m <= 16 && topk == kMaxTopK && h == 3 * kChunkK * RouterCfg<1, true>::kKSplitCTAs) return kFEPathSwapAB;
    return kFEPathWMMA;
}

size_t router_quant_topk_frontend_workspace_bytes(int e) {
    return kFrontendStampsOffsetBase + static_cast<size_t>(4) * 64 * e * 4 + kFrontendStampsBytes;
}

int router_quant_topk_frontend_router_ctas(int m, int h, int e, int topk) {
    if (select_fe_path(m, h, e, topk) == kFEPathCC) {
        int epc = 0, router_ctas = 0;
        cc_plan(e, epc, router_ctas);
        return router_ctas;
    }
    const int groups = (e + kExpertsPerCTA - 1) / kExpertsPerCTA;
    return groups * (m <= 16 ? 4 : 2);
}

void launch_router_quant_topk_frontend(
        const void* hidden, const void* router_weight,
        void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
        void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
        int stamps_on, int l2_persist, int wlayout, int select_in_mega, int zero_row_unrouted, cudaStream_t stream) {
    int* ticket = static_cast<int*>(workspace);
    float* logits = reinterpret_cast<float*>(static_cast<char*>(workspace) + kFrontendStampsOffsetBase);
    unsigned long long* stamps = nullptr;
    if (stamps_on && workspace_bytes >= router_quant_topk_frontend_workspace_bytes(e))
        stamps = reinterpret_cast<unsigned long long*>(
            static_cast<char*>(workspace) + kFrontendStampsOffsetBase + static_cast<size_t>(4) * 64 * e * 4);
    const auto* hp = static_cast<const __nv_bfloat16*>(hidden);
    const auto* wp = static_cast<const __nv_bfloat16*>(router_weight);
    if (mode == 0)
        launch_mode<0>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, stamps, m, h, e, topk, l2_persist, wlayout, select_in_mega, zero_row_unrouted, stream);
    else
        launch_mode<1>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, stamps, m, h, e, topk, l2_persist, wlayout, select_in_mega, zero_row_unrouted, stream);
}
