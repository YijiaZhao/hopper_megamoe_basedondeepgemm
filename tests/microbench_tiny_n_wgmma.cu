// Standalone tensor-pipe microbenchmark for the BM8 swapAB tiny-M tier of the
// H20 fused MegaMoE kernel (sm_90a).
//
// One "stage" = 2 K128 blocks x 256 weight rows x 8 tokens (int8) =
// 256*8*256 = 524288 MACs per SM. The kernel form is: 2 math warpgroups, each
// owning 128 rows (2 m64 halves), issuing per block 8 RS wgmma m64n8k32 s8
// (2 halves x 4 k32 steps) in one commit group, wait<1> after each block.
//
// No memory traffic: all operands are resident in SMEM (B, and A for SS) or in
// registers (A fragments for RS / IMMA). Only the tensor pipe + issue is timed.
//
// Modes (see run script):
//   rs8    : (a) RS  m64n8k32  s8, NWG math WGs x HALVES m64 halves each
//   ss8    : (b) SS  m64n8k32  s8 (A via smem descriptor)
//   rs16   : (c) RS  m64n16k32 s8 (N padded to 16)
//   ss16   :     SS  m64n16k32 s8
//   imma   : (e) mma.sync m16n8k32 s8, 8 warps, A+B in registers
//   immal  : (e2) same, B fragments re-loaded from smem (ld.shared) each k step
//   rs8fp8 : (f) RS  m64n8k32 e4m3 (f32 accum) — MXFP4 path form
//   rs8k   :     RS  m64n8k32 s8 but one commit group per k step (2 wgmma/group), wait<4>
//              (the b304589 inline-s2 loop form)
//
// nvcc -arch=sm_90a -O3 -std=c++17 -o microbench_tiny_n_wgmma microbench_tiny_n_wgmma.cu
// ./microbench_tiny_n_wgmma <mode> <num_math_wgs> <halves_per_wg> [iters]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

enum Mode { RS8 = 0, SS8 = 1, RS16 = 2, SS16 = 3, IMMA = 4, IMMAL = 5, RS8FP8 = 6, RS8K = 7 };

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// K-major 128B-swizzle descriptor (layout_type 1), SBO = 1024 (8 rows x 128 B), LBO = 0.
__device__ __forceinline__ uint64_t make_desc_sw128(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)((0u >> 4) & 0x3FFF) << 16;      // LBO
    desc |= (uint64_t)((1024u >> 4) & 0x3FFF) << 32;   // SBO
    desc |= (uint64_t)1 << 62;                         // SWIZZLE_128B
    return desc;
}

__device__ __forceinline__ void wg_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// RS m64n8k32 s32.s8.s8 : D 4 regs, A 4 regs
__device__ __forceinline__ void wgmma_rs8(uint32_t (&d)[4], const uint32_t (&a)[4], uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
}
// RS m64n8k32 f32.e4m3.e4m3
__device__ __forceinline__ void wgmma_rs8_fp8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
}
// SS m64n8k32 s32.s8.s8
__device__ __forceinline__ void wgmma_ss8(uint32_t (&d)[4], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {%0,%1,%2,%3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}
// RS m64n16k32 s32.s8.s8 : D 8 regs
__device__ __forceinline__ void wgmma_rs16(uint32_t (&d)[8], const uint32_t (&a)[4], uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
}
__device__ __forceinline__ void wgmma_ss16(uint32_t (&d)[8], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}
// legacy mma.sync m16n8k32 s8
__device__ __forceinline__ void mma_imma(uint32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// SMEM layout: B (activations, swapAB "A" smem) : 2 blocks x 8 tokens x 128 B = 2 KB.
//              A tiles for SS: 4 halves x 2 blocks x 64 rows x 128 B = 64 KB.
constexpr int kSmemB = 2 * 8 * 128;
constexpr int kSmemA = 4 * 2 * 64 * 128;

template <int MODE, int NWG, int HALVES>
__global__ void __launch_bounds__(384, 1) bench_kernel(unsigned long long* out, int* sink, int iters) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* smem_b = smem;
    uint8_t* smem_a = smem + kSmemB;
    // fill with small values (content irrelevant for timing)
    for (int i = threadIdx.x; i < kSmemB + kSmemA; i += blockDim.x) smem[i] = (uint8_t)((i * 7) & 3);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wg = warp >> 2;          // 0 = idle "producer" WG, 1..NWG = math WGs
    const bool math = (wg >= 1 && wg <= NWG);
    int acc_sink = 0;

    uint64_t t0 = 0;
    __syncthreads();
    if (threadIdx.x == 0) t0 = gtimer();

    if constexpr (MODE == RS8 || MODE == SS8 || MODE == RS16 || MODE == SS16 || MODE == RS8FP8 || MODE == RS8K) {
        if (math) {
            constexpr int NACC = (MODE == RS16 || MODE == SS16) ? 8 : 4;
            uint32_t acc[HALVES][NACC];
            float accf[HALVES][4];
            uint32_t afrag[HALVES][4][4];
            #pragma unroll
            for (int h = 0; h < HALVES; ++h) {
                #pragma unroll
                for (int j = 0; j < NACC; ++j) acc[h][j] = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) accf[h][j] = 0.f;
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        afrag[h][k][j] = 0x01010101u * (uint32_t)((lane + h + k + j) & 3);
            }
            uint64_t desc_b0 = make_desc_sw128(smem_b);
            uint64_t desc_b1 = make_desc_sw128(smem_b + 1024);
            // SS A: half h of this WG => tile index ((wg-1)*HALVES + h) % 4, block b
            uint64_t desc_a[HALVES][2];
            #pragma unroll
            for (int h = 0; h < HALVES; ++h)
                #pragma unroll
                for (int b = 0; b < 2; ++b)
                    desc_a[h][b] = make_desc_sw128(smem_a + ((((wg - 1) * HALVES + h) & 3) * 2 + b) * 64 * 128);

            #pragma unroll 1
            for (int it = 0; it < iters; ++it) {
                #pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const uint64_t db = b ? desc_b1 : desc_b0;
                    if constexpr (MODE == RS8K) {
                        // b304589 form: per k step: fence, 2 halves, commit; wait<4>
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            wg_fence();
                            #pragma unroll
                            for (int h = 0; h < HALVES; ++h)
                                wgmma_rs8(acc[h], afrag[h][k], db + (uint64_t)((k * 32) >> 4));
                            wg_commit();
                            wg_wait<4>();
                        }
                    } else {
                        wg_fence();
                        #pragma unroll
                        for (int h = 0; h < HALVES; ++h) {
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const uint64_t dbk = db + (uint64_t)((k * 32) >> 4);
                                if constexpr (MODE == RS8)    wgmma_rs8(acc[h], afrag[h][k], dbk);
                                if constexpr (MODE == RS8FP8) wgmma_rs8_fp8(accf[h], afrag[h][k], dbk);
                                if constexpr (MODE == SS8)    wgmma_ss8(acc[h], desc_a[h][b] + (uint64_t)((k * 32) >> 4), dbk);
                                if constexpr (MODE == RS16)   wgmma_rs16(acc[h], afrag[h][k], dbk);
                                if constexpr (MODE == SS16)   wgmma_ss16(acc[h], desc_a[h][b] + (uint64_t)((k * 32) >> 4), dbk);
                            }
                        }
                        wg_commit();
                        wg_wait<1>();
                    }
                }
            }
            wg_wait<0>();
            #pragma unroll
            for (int h = 0; h < HALVES; ++h) {
                #pragma unroll
                for (int j = 0; j < NACC; ++j) acc_sink += (int)acc[h][j];
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc_sink += (int)accf[h][j];
            }
        }
    } else {
        // IMMA: 8 warps (warps 4..11), each owns 32 rows = 2 m16 tiles; 8 k32 steps per block.
        if (warp >= 4 && warp < 12) {
            uint32_t acc[2][4] = {};
            uint32_t afrag[2][4][4];
            uint32_t bfrag[2][8][2];
            #pragma unroll
            for (int t = 0; t < 2; ++t)
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) afrag[t][k][j] = 0x01010101u * (uint32_t)((lane + t + k + j) & 3);
            #pragma unroll
            for (int b = 0; b < 2; ++b)
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    bfrag[b][k][0] = 0x01010101u * (uint32_t)((lane + k) & 3);
                    bfrag[b][k][1] = 0x01010101u * (uint32_t)((lane + k + b) & 3);
                }
            const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(smem_b));
            #pragma unroll 1
            for (int it = 0; it < iters; ++it) {
                #pragma unroll
                for (int b = 0; b < 2; ++b) {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        uint32_t b0, b1;
                        if constexpr (MODE == IMMAL) {
                            // B fragment for (block b, k32 step k): 8 tokens x 32 B; lane -> (n = lane/4, kchunk = lane%4)
                            const uint32_t addr = sb + b * 1024 + (lane >> 2) * 128 + k * 32 + (lane & 3) * 8;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                        } else {
                            b0 = bfrag[b][k][0]; b1 = bfrag[b][k][1];
                        }
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) mma_imma(acc[t], afrag[t][k & 3], b0, b1);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < 2; ++t)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc_sink += (int)acc[t][j];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = gtimer() - t0;
    if (acc_sink == 0x7eadbeef) sink[0] = acc_sink;  // keep accumulators alive
}

typedef void (*kern_t)(unsigned long long*, int*, int);

template <int MODE, int NWG, int HALVES>
kern_t pick() { return bench_kernel<MODE, NWG, HALVES>; }

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <rs8|ss8|rs16|ss16|imma|immal|rs8fp8|rs8k> <nwg 1..3> <halves 1..4> [iters]\n", argv[0]); return 1; }
    const char* mode_s = argv[1];
    int nwg = atoi(argv[2]);
    int halves = atoi(argv[3]);
    int iters = argc > 4 ? atoi(argv[4]) : 1000;
    int mode = -1;
    const char* names[] = {"rs8", "ss8", "rs16", "ss16", "imma", "immal", "rs8fp8", "rs8k"};
    for (int i = 0; i < 8; ++i) if (!strcmp(mode_s, names[i])) mode = i;
    if (mode < 0) { printf("bad mode\n"); return 1; }

    kern_t k = nullptr;
#define SEL(M) \
    if (mode == M) { \
        if (nwg == 1 && halves == 1) k = pick<M, 1, 1>(); \
        if (nwg == 1 && halves == 2) k = pick<M, 1, 2>(); \
        if (nwg == 1 && halves == 4) k = pick<M, 1, 4>(); \
        if (nwg == 2 && halves == 1) k = pick<M, 2, 1>(); \
        if (nwg == 2 && halves == 2) k = pick<M, 2, 2>(); \
        if (nwg == 2 && halves == 4) k = pick<M, 2, 4>(); \
        if (nwg == 3 && halves == 1) k = pick<M, 3, 1>(); \
        if (nwg == 3 && halves == 2) k = pick<M, 3, 2>(); \
    }
    SEL(RS8) SEL(SS8) SEL(RS16) SEL(SS16) SEL(IMMA) SEL(IMMAL) SEL(RS8FP8) SEL(RS8K)
    if (!k) { printf("unsupported nwg/halves combo\n"); return 1; }

    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    const int nsm = prop.multiProcessorCount;
    const size_t smem = kSmemB + kSmemA + 1024;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    unsigned long long* d_out; int* d_sink;
    CK(cudaMalloc(&d_out, nsm * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_sink, sizeof(int)));
    // warmup
    k<<<nsm, 384, smem>>>(d_out, d_sink, 50);
    CK(cudaDeviceSynchronize());
    std::vector<double> per_run;
    for (int r = 0; r < 5; ++r) {
        k<<<nsm, 384, smem>>>(d_out, d_sink, iters);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> h(nsm);
        CK(cudaMemcpy(h.data(), d_out, nsm * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double s = 0; for (auto v : h) s += (double)v;
        per_run.push_back(s / nsm / iters);
    }
    std::sort(per_run.begin(), per_run.end());
    const double ns_stage = per_run[2];  // median
    // work per stage as issued: wgmma modes: NWG*HALVES m64 halves x 2 blocks x K128 x N tokens
    // IMMA modes: 8 warps x 32 rows x 2 blocks x K128 x 8 tokens = 256 rows.
    int n_tok = (mode == RS16 || mode == SS16) ? 16 : 8;
    double rows = (mode == IMMA || mode == IMMAL) ? 256.0 : 64.0 * nwg * halves;
    double macs = rows * n_tok * 256.0;
    // normalised: ns per (256 rows x 8 tokens x K256) equivalent
    double ns_norm = ns_stage * (256.0 * 8.0 * 256.0) / macs;
    double clk = ns_stage * 1.83;
    printf("mode=%-6s nwg=%d halves=%d rows=%.0f ntok=%d : %.1f ns/stage (%.0f clk) ; %.1f MAC/clk/SM ; norm(256x8xK256)=%.1f ns\n",
           mode_s, nwg, halves, rows, n_tok, ns_stage, clk, macs / clk, ns_norm);
    return 0;
}
