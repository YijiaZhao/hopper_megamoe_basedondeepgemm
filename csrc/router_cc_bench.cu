// Standalone CUDA-core router microkernel for tiny M (rows 1..2) on H20 (78 SMs):
//   logits[m x E] = bf16-round( X[m x K] . W[E x K]^T )  (fp32 accumulate)
// with E = 384, K = 3072 (W = 2.36 MB bf16, read once from HBM per launch).
//
// Design: one WARP owns one expert's full K row (kKS = 1) or one half of it (kKS = 2).
// Lane l owns the 16 B K-chunks l, l + 32, ... (12 chunks x 16 B = 192 B per lane per
// expert); the kernel's first instructions are the m activation-row chunk loads (L2-hot,
// they gate the FMAs) immediately followed by ALL weight chunk loads
// (ld.global.nc.L1::no_allocate.v4, pure address arithmetic, no smem, no barrier), so the
// whole CTA's weight footprint (5 experts x 6 KB = 30 KB) is in flight within the first
// ~100 ns. Expert assignment is round-robin across CTAs: warp slot s of CTA b owns experts
// (s * kEPW + j) * gridDim.x + b (j < kEPW), so a 78-CTA grid gives every CTA <= 5 experts.
//
// Accumulation order (fixed): per lane, acc = fma chain over the 8 bf16 of chunk c in
// element order, c = 0..11 (i.e. K = lane*8 + 256*c + i); then the 32 lane partials are
// summed by the xor butterfly (16, 8, 4, 2, 1); K-split partials (kKS = 2) are summed as
// low half + high half; ONE bf16 rounding of the fp32 result (as the FE does).
//
// Variants (see the table in main): kLoad 0 = ld.nc, 1 = + L2::evict_first hint,
// 2 = + L2::256B prefetch, 3 = cp.async.bulk 6 KB/warp into smem then FMA from smem;
// kXMode 0 = activation chunks in registers (per lane), 1 = activation rows in smem (TMA).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -o router_cc_bench router_cc_bench.cu
// Run:   ./router_cc_bench --variant a --rows 1 --iters 100 --stamps 5
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1); } } while (0)

namespace {
constexpr int kH = 3072;               // K
constexpr int kChunkBytes = 16;
constexpr int kChunksPerRow = kH * 2 / (32 * kChunkBytes);   // 12 chunks per lane for the full row
constexpr int kStampSlots = 8;
// stamps per CTA: 0 start / 1 first weight chunk landed (warp 0) / 2 all logits written /
//                 3 flag written / 4 activation chunks landed (warp 0) / 5 all loads issued / 6 smid

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void stamp(unsigned long long* stamps, int slot) {
    if (stamps != nullptr && threadIdx.x == 0) stamps[blockIdx.x * kStampSlots + slot] = globaltimer_ns();
}
__device__ __forceinline__ void stamp_smid(unsigned long long* stamps) {
    if (stamps != nullptr && threadIdx.x == 0) {
        uint32_t v;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
        stamps[blockIdx.x * kStampSlots + 6] = v;
    }
}
__device__ __forceinline__ uint4 ld_nc_na_16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 ld_nc_na_16_hint(const void* p, uint64_t policy) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(policy));
    return v;
}
__device__ __forceinline__ uint4 ld_nc_na_16_pf256(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint64_t l2_evict_first_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
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
__device__ __forceinline__ uint4 ld_shared_16(const void* p) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)));
    return v;
}
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// acc += <8 bf16 of w, 8 bf16 of x> as an fma chain in element order
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

// kWarps: warps per CTA; kEPW: experts per warp; kKS: K-split warps per expert (1 | 2);
// kLoad: weight load mode (see header); kXMode: 0 activations in registers, 1 in smem; kM: rows.
template <int kWarps, int kEPW, int kKS, int kLoad, int kXMode, int kM>
__global__ void __launch_bounds__(kWarps * 32, 1) router_cc_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
        float* __restrict__ logits, int* __restrict__ flags, unsigned long long* stamps, int e, int epoch) {
    static_assert(kChunksPerRow % kKS == 0 && kWarps % kKS == 0, "K split must divide 12 chunks and the warp count");
    constexpr int kChunks = kChunksPerRow / kKS;           // chunks per lane for this warp's K-part
    constexpr int kKPart = kH / kKS;                       // elements
    constexpr int kSlots = kWarps / kKS;                   // expert slots per CTA
    constexpr int kExpertsPerCTA = kSlots * kEPW;
    constexpr int kWBytesPerWarp = kEPW * kKPart * 2;      // smem for kLoad == 3
    __shared__ __align__(128) uint8_t w_s[kLoad == 3 ? kWarps * kWBytesPerWarp : 16];
    __shared__ __align__(128) uint8_t x_s[kXMode == 1 ? kM * kH * 2 : 16];
    __shared__ float part_s[kSlots][kEPW][kM][kKS];
    __shared__ __align__(8) uint64_t w_bar[kWarps];
    __shared__ __align__(8) uint64_t x_bar;
    stamp(stamps, 0);
    stamp_smid(stamps);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = warp / kKS, ks = warp % kKS;
    const int kbase = ks * kKPart;
    int ex[kEPW];
    bool any = false;
    #pragma unroll
    for (int j = 0; j < kEPW; ++j) { ex[j] = (slot * kEPW + j) * gridDim.x + blockIdx.x; any |= ex[j] < e; }

    // probe: an idle last warp (no expert) issues ONE load of the CTA's first weight chunk (same
    // sector warp 0 loads) and stamps its landing -> true "first bytes landed" (slot 1), not
    // ordered behind the issue loop. When every warp has an expert, slot 1 falls back to warp 0's
    // first chunk consumed after the issue loop (an upper bound).
    const bool probe = stamps != nullptr && warp == kWarps - 1 && !any && kLoad != 3;
    if (probe && lane == 0) {
        const uint4 pv = ld_nc_na_16(w + static_cast<int64_t>(blockIdx.x) * kH);
        asm volatile("" :: "r"(pv.x));
        stamps[blockIdx.x * kStampSlots + 1] = globaltimer_ns();
    }
    if constexpr (kLoad == 3 || kXMode == 1) {
        if (threadIdx.x < kWarps) mbar_init(&w_bar[threadIdx.x], 1);
        if (threadIdx.x == 0) mbar_init(&x_bar, 1);
        mbar_fence_init();
        __syncthreads();
    }
    // ---- activation rows: issue first (they gate the FMAs), L2-hot
    uint4 xv[kM][kChunks];
    if constexpr (kXMode == 0) {
        if (any) {
            #pragma unroll
            for (int r = 0; r < kM; ++r)
                #pragma unroll
                for (int c = 0; c < kChunks; ++c)
                    xv[r][c] = ld_nc_na_16(x + static_cast<int64_t>(r) * kH + kbase + (lane + 32 * c) * 8);
        }
    } else {
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(&x_bar, kM * kH * 2);
            #pragma unroll
            for (int r = 0; r < kM; ++r) tma_bulk_g2s(x_s + r * kH * 2, x + static_cast<int64_t>(r) * kH, kH * 2, &x_bar);
        }
    }
    // ---- weights: every chunk of every owned expert in flight now
    uint4 wv[kEPW][kChunks];
    if constexpr (kLoad == 3) {
        if (lane == 0 && any) {
            mbar_arrive_expect_tx(&w_bar[warp], kWBytesPerWarp);
            #pragma unroll
            for (int j = 0; j < kEPW; ++j)
                tma_bulk_g2s(w_s + warp * kWBytesPerWarp + j * kKPart * 2,
                             w + static_cast<int64_t>(min(ex[j], e - 1)) * kH + kbase, kKPart * 2, &w_bar[warp]);
        }
    } else {
        const uint64_t pol = kLoad == 1 ? l2_evict_first_policy() : 0ull;
        #pragma unroll
        for (int j = 0; j < kEPW; ++j) {
            const __nv_bfloat16* wr = w + static_cast<int64_t>(ex[j]) * kH + kbase + lane * 8;
            #pragma unroll
            for (int c = 0; c < kChunks; ++c) {
                wv[j][c] = make_uint4(0u, 0u, 0u, 0u);
                if (ex[j] < e) {
                    if constexpr (kLoad == 1) wv[j][c] = ld_nc_na_16_hint(wr + 256 * c, pol);
                    else if constexpr (kLoad == 2) wv[j][c] = ld_nc_na_16_pf256(wr + 256 * c);
                    else wv[j][c] = ld_nc_na_16(wr + 256 * c);
                }
            }
        }
    }
    stamp(stamps, 5);
    if constexpr (kXMode == 1) {
        mbar_wait_parity(&x_bar, 0u);
        #pragma unroll
        for (int r = 0; r < kM; ++r)
            #pragma unroll
            for (int c = 0; c < kChunks; ++c)
                xv[r][c] = ld_shared_16(x_s + (static_cast<int>(r) * kH + kbase + (lane + 32 * c) * 8) * 2);
    }
    if (stamps != nullptr && threadIdx.x == 0) {   // warp 0 lane 0: activation chunk 0 landed
        asm volatile("" :: "r"(xv[0][0].x));
        stamps[blockIdx.x * kStampSlots + 4] = globaltimer_ns();
    }
    if constexpr (kLoad == 3) {
        if (any) mbar_wait_parity(&w_bar[warp], 0u);
    }
    if (stamps != nullptr && threadIdx.x == 0) {   // warp 0 lane 0: first weight chunk consumed
        if constexpr (kLoad == 3) asm volatile("" ::: "memory");
        else asm volatile("" :: "r"(wv[0][0].x));
        stamps[blockIdx.x * kStampSlots + 7] = globaltimer_ns();
        if (kLoad == 3 || (kWarps - 1) * kEPW * static_cast<int>(gridDim.x) < e) stamps[blockIdx.x * kStampSlots + 1] = stamps[blockIdx.x * kStampSlots + 7];
    }
    // ---- FMA + butterfly per expert
    #pragma unroll
    for (int j = 0; j < kEPW; ++j) {
        float acc[kM];
        #pragma unroll
        for (int r = 0; r < kM; ++r) acc[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            uint4 wc;
            if constexpr (kLoad == 3) wc = ld_shared_16(w_s + warp * kWBytesPerWarp + j * kKPart * 2 + (lane + 32 * c) * 16);
            else wc = wv[j][c];
            #pragma unroll
            for (int r = 0; r < kM; ++r) acc[r] = dot8_bf16(wc, xv[r][c], acc[r]);
        }
        #pragma unroll
        for (int r = 0; r < kM; ++r) {
            acc[r] = warp_sum(acc[r]);
            if (lane == 0) part_s[slot][j][r][ks] = acc[r];
        }
    }
    __syncthreads();
    if (threadIdx.x < kSlots * kEPW * kM) {
        const int r = threadIdx.x % kM, je = threadIdx.x / kM, s = je / kEPW, j = je % kEPW;
        const int exo = (s * kEPW + j) * gridDim.x + blockIdx.x;
        if (exo < e) {
            float v = 0.0f;
            #pragma unroll
            for (int k = 0; k < kKS; ++k) v += part_s[s][j][r][k];
            logits[static_cast<int64_t>(r) * e + exo] = round_bf16(v);
        }
    }
    __syncthreads();
    stamp(stamps, 2);
    if (threadIdx.x == 0) {
        __threadfence();
        st_release_gpu(flags + blockIdx.x, epoch);
    }
    stamp(stamps, 3);
    (void)kExpertsPerCTA;
}

struct Variant {
    const char* name;
    int grid;
    void (*launch)(int grid, const __nv_bfloat16*, const __nv_bfloat16*, float*, int*, unsigned long long*, int, int, cudaStream_t);
    const char* desc;
};
template <int kWarps, int kEPW, int kKS, int kLoad, int kXMode, int kM>
void launch_v(int grid, const __nv_bfloat16* x, const __nv_bfloat16* w, float* logits, int* flags,
              unsigned long long* stamps, int e, int epoch, cudaStream_t st) {
    router_cc_kernel<kWarps, kEPW, kKS, kLoad, kXMode, kM><<<grid, kWarps * 32, 0, st>>>(x, w, logits, flags, stamps, e, epoch);
}
template <int kM>
std::vector<Variant> variants() {
    return {
        {"a",    78, launch_v<8, 1, 1, 0, 0, kM>,  "78 CTA x 8 warps (5 expert warps), 1 expert/warp, ld.nc regs, x regs"},
        {"asx",  78, launch_v<8, 1, 1, 0, 1, kM>,  "a + activation rows via cp.async.bulk into smem"},
        {"b",    78, launch_v<8, 1, 1, 3, 0, kM>,  "a + weights via cp.async.bulk 6 KB/warp into smem, FMA from smem"},
        {"c",    39, launch_v<8, 2, 1, 0, 0, kM>,  "39 CTA x 8 warps (5 active), 2 experts/warp (24 chunks in flight/lane)"},
        {"d1",   78, launch_v<8, 1, 1, 1, 0, kM>,  "a + L2::evict_first cache hint on weight loads"},
        {"d2",   78, launch_v<8, 1, 1, 2, 0, kM>,  "a + L2::256B prefetch on weight loads"},
        {"e",    48, launch_v<8, 1, 1, 0, 0, kM>,  "48 CTA x 8 warps all active, 1 expert/warp (48 KB/SM in flight)"},
        {"f",    96, launch_v<4, 1, 1, 0, 0, kM>,  "96 CTA x 4 warps, 1 expert/warp (2 CTAs on 18 SMs)"},
        {"h",    78, launch_v<10, 1, 2, 0, 0, kM>, "78 CTA x 10 warps: 5 experts x 2 half-K warps (6 chunks/lane)"},
        {"h2",   48, launch_v<16, 1, 2, 0, 0, kM>, "48 CTA x 16 warps: 8 experts x 2 half-K warps"},
        {"c78",  78, launch_v<4, 2, 1, 0, 0, kM>,  "78 CTA x 4 warps (3 active), 2 experts/warp"},
        {"h3",   78, launch_v<15, 1, 3, 0, 0, kM>, "78 CTA x 15 warps: 5 experts x 3 third-K warps (4 chunks/lane)"},
        {"h4",   78, launch_v<20, 1, 4, 0, 0, kM>, "78 CTA x 20 warps: 5 experts x 4 quarter-K warps (3 chunks/lane)"},
        {"h6",   78, launch_v<30, 1, 6, 0, 0, kM>, "78 CTA x 30 warps: 5 experts x 6 warps (2 chunks/lane)"},
        {"h12", 192, launch_v<24, 1, 12, 0, 0, kM>, "192 CTA x 24 warps: 2 experts x 12 warps (1 chunk/lane), 2-3 CTAs per SM"},
        {"h4b",  78, launch_v<20, 1, 4, 3, 0, kM>, "h4 with weights via cp.async.bulk 1.5 KB/warp into smem"},
        {"h4x",  78, launch_v<20, 1, 4, 0, 1, kM>, "h4 with activation rows via cp.async.bulk into smem"},
    };
}

float bf16_to_f(uint16_t b) { uint32_t u = static_cast<uint32_t>(b) << 16; float f; memcpy(&f, &u, 4); return f; }
uint16_t f_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t lsb = (u >> 16) & 1u; u += 0x7fffu + lsb; return static_cast<uint16_t>(u >> 16); }
}  // namespace

int main(int argc, char** argv) {
    std::string vname = "a";
    int rows = 1, iters = 100, warmup = 5, nstamps = 5, grid_override = 0, e = 384, check = 1, flush = 1;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return strcmp(argv[i], k) == 0 && i + 1 < argc; };
        if (arg("--variant")) vname = argv[++i];
        else if (arg("--rows")) rows = atoi(argv[++i]);
        else if (arg("--iters")) iters = atoi(argv[++i]);
        else if (arg("--warmup")) warmup = atoi(argv[++i]);
        else if (arg("--stamps")) nstamps = atoi(argv[++i]);
        else if (arg("--grid")) grid_override = atoi(argv[++i]);
        else if (arg("--check")) check = atoi(argv[++i]);
        else if (arg("--flush")) flush = atoi(argv[++i]);
        else if (strcmp(argv[i], "--list") == 0) { for (auto& v : variants<1>()) printf("%-5s grid %3d  %s\n", v.name, v.grid, v.desc); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (rows != 1 && rows != 2) { fprintf(stderr, "rows must be 1 or 2\n"); return 1; }
    auto vs = rows == 1 ? variants<1>() : variants<2>();
    const Variant* V = nullptr;
    for (auto& v : vs) if (vname == v.name) V = &v;
    if (!V) { fprintf(stderr, "unknown variant %s\n", vname.c_str()); return 1; }
    const int grid = grid_override > 0 ? grid_override : V->grid;

    std::mt19937 rng(20260805);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> hw(static_cast<size_t>(e) * kH), hx(static_cast<size_t>(rows) * kH);
    for (auto& v : hw) v = f_to_bf16(nd(rng) * 0.05f);
    for (auto& v : hx) v = f_to_bf16(nd(rng));
    __nv_bfloat16 *dw, *dx; float* dlog; int* dflags; unsigned long long* dst; uint8_t* scratch;
    CK(cudaMalloc(&dw, hw.size() * 2)); CK(cudaMalloc(&dx, hx.size() * 2));
    CK(cudaMalloc(&dlog, static_cast<size_t>(rows) * e * 4)); CK(cudaMalloc(&dflags, 1024 * 4));
    CK(cudaMalloc(&dst, static_cast<size_t>(grid) * kStampSlots * 8)); CK(cudaMalloc(&scratch, 256u << 20));
    CK(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dflags, 0, 1024 * 4)); CK(cudaMemset(dlog, 0, static_cast<size_t>(rows) * e * 4));
    cudaFuncAttributes fa = {};
    // (attributes are per instantiation; query through a tiny launch-free trick is not available -> print after first launch via cudaFuncGetAttributes on the symbol is variant-specific; skip)
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("== router_cc_bench variant=%s grid=%d rows=%d E=%d K=%d device=%s SMs=%d iters=%d flush=%d ==\n  %s\n",
           V->name, grid, rows, e, kH, prop.name, prop.multiProcessorCount, iters, flush, V->desc);
    (void)fa;

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> times;
    int epoch = 0;
    for (int it = 0; it < warmup + iters; ++it) {
        if (flush) CK(cudaMemsetAsync(scratch, 0, 256u << 20));
        CK(cudaDeviceSynchronize());
        ++epoch;
        CK(cudaEventRecord(e0));
        V->launch(grid, dx, dw, dlog, dflags, nullptr, e, epoch, 0);
        CK(cudaEventRecord(e1));
        CK(cudaDeviceSynchronize());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= warmup) times.push_back(ms * 1e3f);
    }
    CK(cudaGetLastError());
    std::sort(times.begin(), times.end());
    const float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    printf("event us: min %.2f p50 %.2f mean %.2f p90 %.2f max %.2f\n", times.front(), times[times.size() / 2], mean,
           times[times.size() * 9 / 10], times.back());
    // flags check
    std::vector<int> hflags(grid);
    CK(cudaMemcpy(hflags.data(), dflags, grid * 4, cudaMemcpyDeviceToHost));
    int bad_flags = 0;
    for (int b = 0; b < grid; ++b) bad_flags += hflags[b] != epoch;
    // correctness vs double reference
    if (check) {
        std::vector<float> hlog(static_cast<size_t>(rows) * e);
        CK(cudaMemcpy(hlog.data(), dlog, hlog.size() * 4, cudaMemcpyDeviceToHost));
        double max_err = 0.0, max_err_bf = 0.0; int flips = 0;
        for (int r = 0; r < rows; ++r)
            for (int ex = 0; ex < e; ++ex) {
                double ref = 0.0;
                for (int k = 0; k < kH; ++k) ref += static_cast<double>(bf16_to_f(hx[static_cast<size_t>(r) * kH + k])) * bf16_to_f(hw[static_cast<size_t>(ex) * kH + k]);
                const float got = hlog[static_cast<size_t>(r) * e + ex];
                const float ref_bf = bf16_to_f(f_to_bf16(static_cast<float>(ref)));
                max_err = std::max(max_err, std::fabs(static_cast<double>(got) - ref));
                max_err_bf = std::max(max_err_bf, static_cast<double>(std::fabs(got - ref_bf)));
                flips += got != ref_bf;
            }
        printf("check vs fp64 ref: max|logit - ref| %.3e (before bf16), max|logit - bf16(ref)| %.3e, bf16 flips %d / %d, bad flags %d\n",
               max_err, max_err_bf, flips, rows * e, bad_flags);
    }
    // stamps
    if (nstamps > 0) {
        std::vector<unsigned long long> hs(static_cast<size_t>(grid) * kStampSlots);
        const char* names[] = {"start", "w_first_landed(probe)", "logits_written", "flag_written", "x_landed(w0)", "loads_issued", "smid", "w_chunk0_consumed(w0)"};
        const int order[] = {0, 5, 1, 4, 7, 2, 3};
        constexpr int kShown = 7;
        std::vector<std::vector<double>> mins(kShown), meds(kShown), maxs(kShown);
        for (int s = 0; s < nstamps; ++s) {
            if (flush) CK(cudaMemsetAsync(scratch, 0, 256u << 20));
            CK(cudaMemset(dst, 0, hs.size() * 8));
            CK(cudaDeviceSynchronize());
            ++epoch;
            V->launch(grid, dx, dw, dlog, dflags, dst, e, epoch, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hs.data(), dst, hs.size() * 8, cudaMemcpyDeviceToHost));
            unsigned long long t0 = ~0ull;
            for (int b = 0; b < grid; ++b) t0 = std::min(t0, hs[b * kStampSlots + 0]);
            for (int oi = 0; oi < kShown; ++oi) {
                const int slot = order[oi];
                std::vector<double> v;
                for (int b = 0; b < grid; ++b) v.push_back(static_cast<double>(hs[b * kStampSlots + slot] - t0) * 1e-3);
                std::sort(v.begin(), v.end());
                mins[oi].push_back(v.front()); meds[oi].push_back(v[v.size() / 2]); maxs[oi].push_back(v.back());
            }
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        printf("stamps (us from first CTA start; median over %d launches of the per-launch min / median / max across CTAs):\n", nstamps);
        for (int oi = 0; oi < kShown; ++oi)
            printf("  %-24s min %6.2f  med %6.2f  max %6.2f\n", names[order[oi]], med(mins[oi]), med(meds[oi]), med(maxs[oi]));
    }
    return 0;
}
