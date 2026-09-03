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
template <int kMTiles> struct RouterCfg {
    // K-split across CTAs: partial logits go to workspace slice [ks][m][e] and are
    // summed in fixed order by the top-k CTA (deterministic).
    static constexpr int kKSplitCTAs = kMTiles == 1 ? 4 : 2;
    static constexpr int kStages = kMTiles == 1 ? 8 : (kMTiles == 2 ? 6 : 4);
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

template <int kMTiles>
__device__ __forceinline__ void router_role(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        float* __restrict__ logits,          // [m, e] workspace
        int m, int h, int e) {
    using namespace nvcuda;
    constexpr int kStages = RouterCfg<kMTiles>::kStages;
    constexpr int kHRows = RouterCfg<kMTiles>::kHRows;
    constexpr int kStageElems = RouterCfg<kMTiles>::kStageElems;
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
    constexpr int kKSplitCTAs = RouterCfg<kMTiles>::kKSplitCTAs;
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

template <int kMTiles, int kMode>
__global__ void __launch_bounds__(kThreads, 1) router_quant_topk_kernel(
        const __nv_bfloat16* __restrict__ hidden,
        const __nv_bfloat16* __restrict__ router_weight,
        uint8_t* __restrict__ x_bytes, float* __restrict__ x_sf,
        int64_t* __restrict__ topk_idx, float* __restrict__ topk_weights,
        int* __restrict__ ticket, float* __restrict__ logits,
        int m, int h, int e, int topk, int num_router_ctas) {
    if (static_cast<int>(blockIdx.x) >= num_router_ctas) {
        // Quant CTA: quantize this token, then wait for all router CTAs (they are
        // co-resident: <= 24 + 64 CTAs on 132 SMs) and run top-k for this token.
        const int token = blockIdx.x - num_router_ctas;
        quant_role<kMode>(hidden, x_bytes, x_sf, token, h);
        if (threadIdx.x < 32) {
            if (threadIdx.x == 0) {
                while (*reinterpret_cast<volatile int*>(ticket) < num_router_ctas) __nanosleep(200);
                __threadfence();
            }
            __syncwarp();
            topk_softmax_token(logits, topk_idx, topk_weights, token, m, e, topk, RouterCfg<kMTiles>::kKSplitCTAs);
            // Last token CTA resets the counters for the next launch.
            if (threadIdx.x == 0 && atomicAdd(ticket + 1, 1) == m - 1) {
                ticket[1] = 0;
                __threadfence();
                ticket[0] = 0;
            }
        }
        return;
    }
    router_role<kMTiles>(hidden, router_weight, logits, m, h, e);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(ticket, 1);
}

template <int kMTiles, int kMode>
void launch(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
            int64_t* idx, float* wts, int* ticket, float* logits,
            int m, int h, int e, int topk, cudaStream_t stream) {
    const int router_ctas = ((e + kExpertsPerCTA - 1) / kExpertsPerCTA) * RouterCfg<kMTiles>::kKSplitCTAs;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(router_quant_topk_kernel<kMTiles, kMode>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, RouterCfg<kMTiles>::kDynSmemBytes);
        attr_set = true;
    }
    router_quant_topk_kernel<kMTiles, kMode><<<router_ctas + m, kThreads, RouterCfg<kMTiles>::kDynSmemBytes, stream>>>(
        hidden, w, x, sf, idx, wts, ticket, logits, m, h, e, topk, router_ctas);
}

template <int kMode>
void launch_mode(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
                 int64_t* idx, float* wts, int* ticket, float* logits,
                 int m, int h, int e, int topk, cudaStream_t stream) {
    if (m <= 16) launch<1, kMode>(hidden, w, x, sf, idx, wts, ticket, logits, m, h, e, topk, stream);
    else if (m <= 32) launch<2, kMode>(hidden, w, x, sf, idx, wts, ticket, logits, m, h, e, topk, stream);
    else launch<4, kMode>(hidden, w, x, sf, idx, wts, ticket, logits, m, h, e, topk, stream);
}

}  // namespace

void launch_router_quant_topk_frontend(
        const void* hidden, const void* router_weight,
        void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
        void* workspace, int m, int h, int e, int topk, int mode,
        cudaStream_t stream) {
    int* ticket = static_cast<int*>(workspace);
    float* logits = reinterpret_cast<float*>(static_cast<char*>(workspace) + 256);
    const auto* hp = static_cast<const __nv_bfloat16*>(hidden);
    const auto* wp = static_cast<const __nv_bfloat16*>(router_weight);
    if (mode == 0)
        launch_mode<0>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, m, h, e, topk, stream);
    else
        launch_mode<1>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, m, h, e, topk, stream);
}
