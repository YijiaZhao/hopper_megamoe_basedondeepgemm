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
#include <deep_gemm/impls/fable_frontend_device.cuh>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

namespace {
// Device code (router WMMA, tiny top-k, quantisation) lives in
// deep_gemm/include/deep_gemm/impls/fable_frontend_device.cuh so that the fused
// MegaMoE kernel (DG_FP4_FUSE_FE) runs the very same math; this file keeps the
// launch, the CTA roles and the legacy per-warp top-k.
using namespace deep_gemm::fable_fe;

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
template <int kKSplit, int kPerLane, int kTopK>
__device__ __forceinline__ void topk_tiny_cta(
        const float* __restrict__ logits, int64_t* __restrict__ topk_idx,
        float* __restrict__ topk_weights, unsigned long long* stamps, int t, int m, int e) {
    __shared__ uint32_t key_s[kMaxExperts];
    topk_softmax_token_tiny<kKSplit, kPerLane, kTopK>(logits, topk_idx, topk_weights, stamps, t, m, e,
                                                      static_cast<int>(threadIdx.x), key_s, CtaSync{});
}

// Quant CTA wrapper (static smem for the mode-1 warp maxima).
template <int kMode>
__device__ __forceinline__ void quant_cta(
        const __nv_bfloat16* __restrict__ hidden, uint8_t* __restrict__ x_bytes,
        float* __restrict__ x_sf, int token, int h) {
    __shared__ float smem_warp_max[kThreads / 32];
    quant_role<kMode>(hidden, x_bytes, x_sf, token, h, static_cast<int>(threadIdx.x), smem_warp_max, CtaSync{});
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
        int m, int h, int e, int topk, int num_router_ctas, int w_hint, int pdl_mode) {
    stamp(stamps, 0);
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

template <int kMTiles, int kMode, bool kTiny>
void launch(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
            int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
            int m, int h, int e, int topk, int l2_persist, int pdl_mode, cudaStream_t stream) {
    using Cfg = RouterCfg<kMTiles, kTiny>;
    const int router_ctas = ((e + kExpertsPerCTA - 1) / kExpertsPerCTA) * Cfg::kKSplitCTAs;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(router_quant_topk_kernel<kMTiles, kMode, kTiny>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kDynSmemBytes);
        attr_set = true;
    }
    const size_t w_bytes = static_cast<size_t>(e) * h * sizeof(__nv_bfloat16);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(router_ctas + m);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = Cfg::kDynSmemBytes;
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
    cudaLaunchKernelEx(&cfg, router_quant_topk_kernel<kMTiles, kMode, kTiny>,
                       hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, router_ctas, w_hint, pdl_mode);
}

template <int kMode>
void launch_mode(const __nv_bfloat16* hidden, const __nv_bfloat16* w, uint8_t* x, float* sf,
                 int64_t* idx, float* wts, int* ticket, float* logits, unsigned long long* stamps,
                 int m, int h, int e, int topk, bool tiny, int l2_persist, int pdl_mode, cudaStream_t stream) {
    if (m <= 16 && tiny && topk == kMaxTopK) launch<1, kMode, true>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, stream);
    else if (m <= 16) launch<1, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, stream);
    else if (m <= 32) launch<2, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, stream);
    else launch<4, kMode, false>(hidden, w, x, sf, idx, wts, ticket, logits, stamps, m, h, e, topk, l2_persist, pdl_mode, stream);
}

}  // namespace

size_t router_quant_topk_frontend_workspace_bytes(int e) {
    return kFrontendStampsOffsetBase + static_cast<size_t>(4) * 64 * e * 4 + kFrontendStampsBytes;
}

void launch_router_quant_topk_frontend(
        const void* hidden, const void* router_weight,
        void* x_bytes, void* x_sf, void* topk_idx, void* topk_weights,
        void* workspace, size_t workspace_bytes, int m, int h, int e, int topk, int mode,
        int tiny, int stamps_on, int l2_persist, int pdl_mode, cudaStream_t stream) {
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
                       ticket, logits, stamps, m, h, e, topk, use_tiny, l2_persist, pdl_mode, stream);
    else
        launch_mode<1>(hp, wp, static_cast<uint8_t*>(x_bytes), static_cast<float*>(x_sf),
                       static_cast<int64_t*>(topk_idx), static_cast<float*>(topk_weights),
                       ticket, logits, stamps, m, h, e, topk, use_tiny, l2_persist, pdl_mode, stream);
}
