// Standalone tensor-pipe microbenchmark for the BM8 swapAB tiny-M tier of the
// H20 fused MegaMoE kernel (sm_90a).
//
// One "stage" = 2 K128 blocks x 256 weight rows x 8 tokens (int8) =
// 256*8*256 = 524288 MACs per SM. Kernel form: 2 math warpgroups, each owning
// 128 rows (2 m64 halves), issuing per block 8 RS wgmma m64n8k32 s8 (2 halves x
// 4 k32 steps) in one commit group, wait<1> after each block, A fragments
// re-decoded per block from packed int4 smem into a 2-buffer register set.
//
// No memory traffic: all operands are resident in SMEM (B, and A for SS) or in
// registers (A fragments for RS / IMMA). Only tensor pipe + issue + decode is timed.
//
// Modes:
//   rs8    : (a) RS  m64n8k32  s8, constant A regs (pure pipe), NWG math WGs x HALVES each
//   rs8k   :     same, one commit group per k step (2 wgmma/group), wait<4> (b304589 form)
//   ss8    : (b) SS  m64n8k32  s8 (A via smem descriptor)
//   rs16   : (c) RS  m64n16k32 s8 (N padded to 16);  ss16: SS m64n16k32
//   imma   : (e) mma.sync m16n8k32 s8, 8 warps, A+B in registers; immal: B via ld.shared
//   rs8fp8 : (f) RS  m64n8k32 e4m3 (f32 accum) — MXFP4 path form
//   rs8d   : (a-faithful) kernel 2-buffer loop: issue(b) ; wait<1> ; decode(b^1) ; ...
//              decode = QoQ inline-s2 (4 LDS.128 + 4 LDS.32 + ~100 ALU per thread per block)
//   rs8t   :     3-buffer software pipeline: issue(j) ; wait<2> ; decode(j+1) — two groups
//              stay in flight while the warps decode
//   rs8a   :     rs8 (constant A) + the rs8d decode work into registers that do NOT feed
//              the wgmma — does warp-side work overlap RS wgmma at all?
//   ss8d   :     decode -> int8 smem A tile (st.shared.v4) + fence.proxy.async + SS wgmma
//   dec    :     decode only (rs8d without the wgmma)
//   ss8p   :     decode OFFLOADED: writer warps decode int8 into a double-buffered SW128 smem A
//              tile (conflict-free st.shared.v4 + fence.proxy.async), named-barrier handoff,
//              math WGs only issue SS m64n8k32 + wait<1>. FLAGS 16 = all 12 warps decode
//              (else only the 4 producer warps of WG0); nwg=2 halves=2 only
//   ss8u   :     3-tile offload, deferred affine: WG0 decodes RAW nibble codes to u8 (AND/SHF only)
//              into 3 rotating SW128 tiles; math WGs issue SS m64n8k32 s32.u8.s8 into a per-block
//              accumulator and fold acc_main += s2*acc_blk + (a-128)*colsum(B) in int32 after the
//              wait (exact integers -> bit-identical to the inline-s2 form). Two tiles ahead, so the
//              writers' decode overlaps the math WGs' (blocking) IGMMA issue.
//   Per-phase clock64 stamps (lane 0 of each warp) are printed for rs8d / ss8p / ss8u:
//     rs8d math: issue | wait | decode      ss8p/ss8u math: barF | issue | wait+promote | barE
//     writers: barE | decode+store | fence | barF
//   DEC=<n> env: repeat the decode n times per block (rs8d/rs8t/ss8d/dec/rs8a), default 1
//   FLAGS=<bits> (compile-time instantiations, see main): 1 = decode without LDS (ALU only)
//        2 = ss8d: skip fence.proxy.async (timing only)   4 = ss8d: conflict-free stores
//        8 = phase-offset: math WG2 starts ~200 ns late (WG3 ~400 ns)
//
// nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -o mb microbench_tiny_n_wgmma.cu
// ./mb <mode> <num_math_wgs> <halves_per_wg> [iters]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <type_traits>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

enum Mode { RS8 = 0, SS8 = 1, RS16 = 2, SS16 = 3, IMMA = 4, IMMAL = 5, RS8FP8 = 6, RS8K = 7,
            RS8D = 8, SS8D = 9, DEC = 10, RS8A = 11, RS8T = 12, SS8P = 13, SS8U = 14, NUM_MODES = 15 };
static const char* kModeNames[NUM_MODES] = {"rs8", "ss8", "rs16", "ss16", "imma", "immal", "rs8fp8", "rs8k",
                                            "rs8d", "ss8d", "dec", "rs8a", "rs8t", "ss8p", "ss8u"};

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// K-major 128B-swizzle descriptor (layout_type 1), SBO = 1024 (8 rows x 128 B), LBO = 0.
__device__ __forceinline__ uint64_t make_desc_sw128(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)((1024u >> 4) & 0x3FFF) << 32;   // SBO
    desc |= (uint64_t)1 << 62;                         // SWIZZLE_128B
    return desc;
}

__device__ __forceinline__ void nbar_sync(int id, int n)   { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wg_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void wgmma_rs8(uint32_t (&d)[4], const uint32_t (&a)[4], uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
}
// SS m64n8k32 s32.u8.s8 (A unsigned raw codes)
__device__ __forceinline__ void wgmma_ss8_u8(uint32_t (&d)[4], uint64_t desc_a, uint64_t desc_b, bool scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8 {%0,%1,%2,%3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"((int)scale_d));
}
__device__ __forceinline__ void wgmma_rs8_fp8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(1));
}
__device__ __forceinline__ void wgmma_ss8(uint32_t (&d)[4], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {%0,%1,%2,%3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}
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
__device__ __forceinline__ void mma_imma(uint32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// SMEM: B (activations) 2 blocks x 8 tokens x 128 B; A tiles (SS) 4 halves x 2 blocks x 8 KB;
//       packed int4 weights 2 stages x 256 rows x 80 B (kernel layout: 64 B nibbles + scale words).
constexpr int kSmemB = 2 * 8 * 128;
constexpr int kSmemA = 4 * 3 * 64 * 128;   // 4 m64 tiles x up to 3 rotating buffers x 8 KB
constexpr int kSmemP = 2 * 256 * 80;

// QoQ inline-s2 decode of one m64 half for one K128 block: rows (r0, r0+8) x 4 k32 steps
// -> f[k][0..3], exactly decode_stage_rf in sm90_fp4_mega_moe_h20_fused_body.inl.
template <bool NO_LDS>
__device__ __forceinline__ void decode_half(uint32_t (&f)[4][4], const uint8_t* packed_rows, int row0, int col, uint32_t salt) {
    uint4 w0, w1; uint32_t sw0, sw1;
    if constexpr (NO_LDS) {
        w0 = make_uint4(salt, salt * 3u, salt * 5u, salt * 7u);
        w1 = make_uint4(salt ^ 0x11u, salt ^ 0x33u, salt ^ 0x55u, salt ^ 0x77u);
        sw0 = (salt & 0x7u) | 0x0300u; sw1 = ((salt >> 3) & 0x7u) | 0x0500u;
    } else {
        w0 = *reinterpret_cast<const uint4*>(packed_rows + row0 * 80 + col * 16);
        w1 = *reinterpret_cast<const uint4*>(packed_rows + (row0 + 8) * 80 + col * 16);
        sw0 = *reinterpret_cast<const uint32_t*>(packed_rows + row0 * 80 + 64);
        sw1 = *reinterpret_cast<const uint32_t*>(packed_rows + (row0 + 8) * 80 + 64);
    }
    // opaque: the ALU below must be redone per call (DEC reps / dec mode)
    asm volatile("" : "+r"(w0.x), "+r"(w0.y), "+r"(w0.z), "+r"(w0.w), "+r"(w1.x), "+r"(w1.y), "+r"(w1.z), "+r"(w1.w), "+r"(sw0), "+r"(sw1));
    const uint32_t s2_0 = sw0 & 0xffu, s2_1 = sw1 & 0xffu;
    const uint32_t a4_0 = (0x80u - ((sw0 >> 8u) & 0xffu) * s2_0) * 0x01010101u;
    const uint32_t a4_1 = (0x80u - ((sw1 >> 8u) & 0xffu) * s2_1) * 0x01010101u;
    const uint32_t wr0[4] = {w0.x, w0.y, w0.z, w0.w};
    const uint32_t wr1[4] = {w1.x, w1.y, w1.z, w1.w};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        f[k][0] = (((wr0[k] >> 4) & 0x0f0f0f0fu) * s2_0 + a4_0) ^ 0x80808080u;
        f[k][1] = (((wr1[k] >> 4) & 0x0f0f0f0fu) * s2_1 + a4_1) ^ 0x80808080u;
        f[k][2] = ((wr0[k] & 0x0f0f0f0fu) * s2_0 + a4_0) ^ 0x80808080u;
        f[k][3] = ((wr1[k] & 0x0f0f0f0fu) * s2_1 + a4_1) ^ 0x80808080u;
    }
}

// Offloaded decode: one work item = 16 B packed (32 int4 of one row) -> 32 int8 -> two 16 B
// chunks of the SW128 K-major smem A tile (tile t = row/64, 64 rows x 128 B, chunk ^= row&7).
__device__ __forceinline__ void decode_item_to_tile(uint32_t tile_base, const uint8_t* packed_rows, int row, int c16) {
    const uint4 w = *reinterpret_cast<const uint4*>(packed_rows + row * 80 + c16 * 16);
    const uint32_t sw = *reinterpret_cast<const uint32_t*>(packed_rows + row * 80 + 64);
    const uint32_t s2 = sw & 0xffu;
    const uint32_t a4 = (0x80u - ((sw >> 8u) & 0xffu) * s2) * 0x01010101u;
    const uint32_t wr[4] = {w.x, w.y, w.z, w.w};
    uint32_t lo[4], hi[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        lo[k] = ((wr[k] & 0x0f0f0f0fu) * s2 + a4) ^ 0x80808080u;
        hi[k] = (((wr[k] >> 4) & 0x0f0f0f0fu) * s2 + a4) ^ 0x80808080u;
    }
    const int rr = row & 63;
    const uint32_t base = tile_base + (row >> 6) * 3 * 8192 + rr * 128;   // tile t at (t*3 + p) * 8192
    const uint32_t a_lo = base + (((2 * c16) ^ (rr & 7)) * 16);
    const uint32_t a_hi = base + (((2 * c16 + 1) ^ (rr & 7)) * 16);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(a_lo), "r"(lo[0]), "r"(lo[1]), "r"(lo[2]), "r"(lo[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(a_hi), "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]) : "memory");
}

// Raw-code variant: nibble -> u8 code only (no affine), 3 ALU per word pair.
__device__ __forceinline__ void decode_item_raw_to_tile(uint32_t tile_base, const uint8_t* packed_rows, int row, int c16) {
    const uint4 w = *reinterpret_cast<const uint4*>(packed_rows + row * 80 + c16 * 16);
    const uint32_t wr[4] = {w.x, w.y, w.z, w.w};
    uint32_t lo[4], hi[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) { lo[k] = wr[k] & 0x0f0f0f0fu; hi[k] = (wr[k] >> 4) & 0x0f0f0f0fu; }
    const int rr = row & 63;
    const uint32_t base = tile_base + (row >> 6) * 3 * 8192 + rr * 128;   // tile t at (t*3 + p) * 8192
    const uint32_t a_lo = base + (((2 * c16) ^ (rr & 7)) * 16);
    const uint32_t a_hi = base + (((2 * c16 + 1) ^ (rr & 7)) * 16);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(a_lo), "r"(lo[0]), "r"(lo[1]), "r"(lo[2]), "r"(lo[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(a_hi), "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]) : "memory");
}

template <int MODE, int NWG, int HALVES, int FLAGS>
__global__ void __launch_bounds__(384, 1) bench_kernel(unsigned long long* out, unsigned long long* out_ph, int* sink, int iters, int dec_rep) {
    unsigned long long ph[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #define STAMP(i, expr) do { const unsigned long long _t0 = clock64(); expr; ph[i] += clock64() - _t0; } while (0)
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* smem_b = smem;
    uint8_t* smem_a = smem + kSmemB;
    uint8_t* smem_p = smem + kSmemB + kSmemA;
    for (int i = threadIdx.x; i < kSmemB + kSmemA + kSmemP; i += blockDim.x) smem[i] = (uint8_t)((i * 7) & 3);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wg = warp >> 2;          // 0 = idle "producer" WG, 1..NWG = math WGs
    const bool math = (wg >= 1 && wg <= NWG);
    int acc_sink = 0;

    uint64_t t0 = 0;
    __syncthreads();
    if (threadIdx.x == 0) t0 = gtimer();
    if constexpr (FLAGS & 8) { if (wg >= 2) __nanosleep(200 * (wg - 1)); }

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
            const uint64_t desc_b0 = make_desc_sw128(smem_b);
            const uint64_t desc_b1 = make_desc_sw128(smem_b + 1024);
            uint64_t desc_a[HALVES][2];
            #pragma unroll
            for (int h = 0; h < HALVES; ++h)
                #pragma unroll
                for (int b = 0; b < 2; ++b)
                    desc_a[h][b] = make_desc_sw128(smem_a + ((((wg - 1) * HALVES + h) & 3) * 3 + b) * 64 * 128);

            #pragma unroll 1
            for (int it = 0; it < iters; ++it) {
                #pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const uint64_t db = b ? desc_b1 : desc_b0;
                    if constexpr (MODE == RS8K) {
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
    } else if constexpr (MODE == RS8D || MODE == SS8D || MODE == DEC || MODE == RS8A || MODE == RS8T) {
        if (math) {
            constexpr int NBUF = (MODE == RS8T) ? 3 : 2;
            constexpr bool NO_LDS = FLAGS & 1;
            uint32_t acc[HALVES][4];
            uint32_t fbuf[NBUF][HALVES][4][4];
            uint32_t fconst[HALVES][4][4];   // RS8A: wgmma A operands (constant), decode goes to fbuf
            #pragma unroll
            for (int h = 0; h < HALVES; ++h) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[h][j] = 0;
                #pragma unroll
                for (int b = 0; b < NBUF; ++b)
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) fbuf[b][h][k][j] = 0;
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) fconst[h][k][j] = 0x01010101u * (uint32_t)((lane + h + k + j) & 3);
            }
            const uint64_t desc_b0 = make_desc_sw128(smem_b);
            const uint64_t desc_b1 = make_desc_sw128(smem_b + 1024);
            uint64_t desc_a[HALVES][2];
            uint32_t tile_addr[HALVES][2];
            #pragma unroll
            for (int h = 0; h < HALVES; ++h)
                #pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint8_t* t = smem_a + ((((wg - 1) * HALVES + h) & 3) * 3 + b) * 64 * 128;
                    desc_a[h][b] = make_desc_sw128(t);
                    tile_addr[h][b] = static_cast<uint32_t>(__cvta_generic_to_shared(t)) +
                                      ((FLAGS & 4) ? (threadIdx.x & 127) * 16 : (threadIdx.x & 127) * 64);
                }
            const int wg_n = ((wg - 1) * HALVES * 64) & 255;
            const int r0 = lane >> 2;          // kernel r_0
            const int col = lane & 3;          // kernel col_idx
            const int wrow = (warp & 3) * 16;  // warp's 16-row slab within the m64 half
            // decode fragment buffer `b` (both halves) from packed stage `st`
            auto decode = [&](auto bconst, int st) {
                constexpr int b = decltype(bconst)::value;
                const uint8_t* packed_rows = smem_p + st * (256 * 80);
                asm volatile("" ::: "memory");
                #pragma unroll
                for (int h = 0; h < HALVES; ++h)
                    decode_half<NO_LDS>(fbuf[b][h], packed_rows, wg_n + h * 64 + wrow + r0, col, (uint32_t)(lane + h * 8 + b * 16 + st * 32));
            };
            auto fence_buf = [&](auto bconst) {   // the kernel's fence_frag: definition point after the wait
                constexpr int b = decltype(bconst)::value;
                #pragma unroll
                for (int h = 0; h < HALVES; ++h)
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        asm volatile("" : "+r"(fbuf[b][h][k][0]), "+r"(fbuf[b][h][k][1]), "+r"(fbuf[b][h][k][2]), "+r"(fbuf[b][h][k][3]));
            };
            auto store_tile = [&](auto bconst) {  // SS design: decoded int8 of buffer b -> smem A tile b
                constexpr int b = decltype(bconst)::value;
                #pragma unroll
                for (int h = 0; h < HALVES; ++h)
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(tile_addr[h][b] + k * ((FLAGS & 4) ? 2048 : 16)),
                                     "r"(fbuf[b][h][k][0]), "r"(fbuf[b][h][k][1]), "r"(fbuf[b][h][k][2]), "r"(fbuf[b][h][k][3]) : "memory");
                if constexpr (!(FLAGS & 2)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            };
            auto issue = [&](auto bconst, int blk) {   // buffer b, activation block blk (0/1)
                constexpr int b = decltype(bconst)::value;
                const uint64_t db = blk ? desc_b1 : desc_b0;
                wg_fence();
                #pragma unroll
                for (int h = 0; h < HALVES; ++h)
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint64_t dbk = db + (uint64_t)((k * 32) >> 4);
                        if constexpr (MODE == RS8D || MODE == RS8T) wgmma_rs8(acc[h], fbuf[b][h][k], dbk);
                        if constexpr (MODE == RS8A) wgmma_rs8(acc[h], fconst[h][k], dbk);
                        if constexpr (MODE == SS8D) wgmma_ss8(acc[h], desc_a[h][blk] + (uint64_t)((k * 32) >> 4), dbk);
                    }
                wg_commit();
            };
            auto consume = [&](auto bconst) {  // DEC / RS8A: fold the decoded registers into the per-thread sink
                constexpr int b = decltype(bconst)::value;
                #pragma unroll
                for (int h = 0; h < HALVES; ++h)
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        acc_sink ^= (int)(fbuf[b][h][k][0] ^ (fbuf[b][h][k][1] << 1) ^ (fbuf[b][h][k][2] << 2) ^ (fbuf[b][h][k][3] << 3));
            };
            using I0 = std::integral_constant<int, 0>;
            using I1 = std::integral_constant<int, 1>;
            using I2 = std::integral_constant<int, 2>;

            if constexpr (MODE == RS8T) {
                // 3-buffer pipeline, 6 blocks (3 stages) per outer iteration:
                //   issue(j%3, blk) ; wait<2> [retires j-2 -> buffer (j+1)%3 free] ; decode((j+1)%3)
                for (int r = 0; r < dec_rep; ++r) decode(I0{}, 0);
                #pragma unroll 1
                for (int it = 0; it < iters / 3; ++it) {
                    issue(I0{}, 0); wg_wait<2>(); fence_buf(I1{}); for (int r = 0; r < dec_rep; ++r) decode(I1{}, 0);
                    issue(I1{}, 1); wg_wait<2>(); fence_buf(I2{}); for (int r = 0; r < dec_rep; ++r) decode(I2{}, 1);
                    issue(I2{}, 0); wg_wait<2>(); fence_buf(I0{}); for (int r = 0; r < dec_rep; ++r) decode(I0{}, 1);
                    issue(I0{}, 1); wg_wait<2>(); fence_buf(I1{}); for (int r = 0; r < dec_rep; ++r) decode(I1{}, 0);
                    issue(I1{}, 0); wg_wait<2>(); fence_buf(I2{}); for (int r = 0; r < dec_rep; ++r) decode(I2{}, 1);
                    issue(I2{}, 1); wg_wait<2>(); fence_buf(I0{}); for (int r = 0; r < dec_rep; ++r) decode(I0{}, 0);
                }
                wg_wait<0>();
            } else {
                for (int r = 0; r < dec_rep; ++r) decode(I0{}, 0);
                if constexpr (MODE == SS8D) store_tile(I0{});
                #pragma unroll 1
                for (int it = 0; it < iters; ++it) {
                    const int st = it & 1;
                    // kernel loop body (2-buffer form):
                    //   issue(b0,f0); wait<1> [retires prev b1 -> f1 free]; decode f1; issue(b1,f1);
                    //   wait<1> [retires b0 -> f0 free]; [next stage barrier]; decode f0 (next stage)
                    if constexpr (MODE != DEC) { STAMP(0, issue(I0{}, 0)); STAMP(1, wg_wait<1>()); }
                    if constexpr (MODE == DEC || MODE == RS8A) consume(I0{});
                    fence_buf(I1{});
                    STAMP(2, for (int r = 0; r < dec_rep; ++r) decode(I1{}, st));
                    if constexpr (MODE == SS8D) store_tile(I1{});
                    if constexpr (MODE != DEC) { STAMP(0, issue(I1{}, 1)); STAMP(1, wg_wait<1>()); }
                    if constexpr (MODE == DEC || MODE == RS8A) consume(I1{});
                    fence_buf(I0{});
                    STAMP(2, for (int r = 0; r < dec_rep; ++r) decode(I0{}, st ^ 1));
                    if constexpr (MODE == SS8D) store_tile(I0{});
                }
                if constexpr (MODE != DEC) wg_wait<0>();
            }
            #pragma unroll
            for (int h = 0; h < HALVES; ++h)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc_sink += (int)acc[h][j];
            consume(I0{}); consume(I1{});
            if constexpr (NBUF == 3) consume(I2{});
        }
    } else if constexpr (MODE == SS8P) {
        static_assert(NWG == 2 && HALVES == 2, "ss8p: 2 math WGs x 2 halves");
        constexpr bool ALL_DECODE = FLAGS & 16;
        constexpr int kF0 = 1, kF1 = 2, kE0 = 3, kE1 = 4;   // named barriers: tile full / tile empty
        const bool writer = ALL_DECODE || wg == 0;
        const uint32_t smem_a_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_a));
        // writer round: decode block `blk` parity tile p = blk&1 from packed stage st
        auto write_tile = [&](int p, int st) {
            const uint8_t* packed_rows = smem_p + st * (256 * 80);
            const uint32_t tile_base = smem_a_u32 + p * 8192;   // tile t at (t*3 + p) * 8192
            asm volatile("" ::: "memory");
            for (int r = 0; r < dec_rep; ++r) {
                if constexpr (ALL_DECODE) {
                    #pragma unroll
                    for (int i = threadIdx.x; i < 1024; i += 384)
                        decode_item_to_tile(tile_base, packed_rows, i >> 2, i & 3);
                } else {
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const int i = threadIdx.x + q * 128;
                        decode_item_to_tile(tile_base, packed_rows, i >> 2, i & 3);
                    }
                }
            }
            if constexpr (!(FLAGS & 2)) STAMP(6, asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"));
        };
        uint32_t acc[2][4] = {};
        const uint64_t desc_b0 = make_desc_sw128(smem_b);
        const uint64_t desc_b1 = make_desc_sw128(smem_b + 1024);
        uint64_t desc_a[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int b = 0; b < 2; ++b)
                desc_a[h][b] = make_desc_sw128(smem_a + ((((wg - 1) * 2 + h) & 3) * 3 + b) * 8192);
        auto issue = [&](int p) {
            const uint64_t db = p ? desc_b1 : desc_b0;
            wg_fence();
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    wgmma_ss8(acc[h], desc_a[h][p] + (uint64_t)((k * 32) >> 4), db + (uint64_t)((k * 32) >> 4));
            wg_commit();
        };
        // prologue: tile 0 (block 0)
        if (writer) { write_tile(0, 0); if constexpr (!ALL_DECODE) nbar_arrive(kF0, 384); }
        #pragma unroll 1
        for (int it = 0; it < iters; ++it) {
            const int st = it & 1;
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                // math: wait tile p full, issue, retire the previous group (tile p^1), release tile p^1
                // writers: wait tile p^1 empty, decode the next block into it, publish
                if constexpr (ALL_DECODE) {
                    STAMP(0, nbar_sync(p ? kF1 : kF0, 384));
                    if (math) { STAMP(1, issue(p)); STAMP(2, wg_wait<1>()); }
                    STAMP(3, nbar_sync(p ? kE0 : kE1, 384));
                    STAMP(5, write_tile(p ^ 1, st ^ p));
                } else {
                    if (math) {
                        STAMP(0, nbar_sync(p ? kF1 : kF0, 384));
                        STAMP(1, issue(p)); STAMP(2, wg_wait<1>());
                        STAMP(3, nbar_arrive(p ? kE0 : kE1, 384));
                    } else {
                        STAMP(4, nbar_sync(p ? kE0 : kE1, 384));
                        STAMP(5, write_tile(p ^ 1, st ^ p));
                        STAMP(7, nbar_arrive(p ? kF0 : kF1, 384));
                    }
                }
            }
        }
        if constexpr (!ALL_DECODE) { if (math) nbar_sync(kF0, 384); }   // consume the writers' last publish
        if (math) {
            wg_wait<0>();
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc_sink += (int)acc[h][j];
        }
    } else if constexpr (MODE == SS8U) {
        static_assert(NWG == 2 && HALVES == 2, "ss8u: 2 math WGs x 2 halves");
        // named barriers: F[t] tile t full (1..3), E[t] tile t empty (4..6)
        const uint32_t smem_a_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_a));
        const bool writer = (wg == 0);
        auto write_tile = [&](int t, int st) {
            const uint8_t* packed_rows = smem_p + st * (256 * 80);
            const uint32_t tile_base = smem_a_u32 + t * 8192;
            asm volatile("" ::: "memory");
            for (int r = 0; r < dec_rep; ++r) {
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const int i = threadIdx.x + q * 128;
                    decode_item_raw_to_tile(tile_base, packed_rows, i >> 2, i & 3);
                }
            }
            if constexpr (!(FLAGS & 2)) STAMP(6, asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"));
        };
        uint32_t acc_main[2][4] = {};
        uint32_t acc_blk[2][2][4] = {};   // [parity][half][4]
        const uint64_t desc_b0 = make_desc_sw128(smem_b);
        const uint64_t desc_b1 = make_desc_sw128(smem_b + 1024);
        uint64_t desc_a[2][3];
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int t = 0; t < 3; ++t)
                desc_a[h][t] = make_desc_sw128(smem_a + ((((wg - 1) * 2 + h) & 3) * 3 + t) * 8192);
        // per-row affine (s2, a-128) for the 2 rows this thread's accumulator covers, per block parity
        // (in the kernel: 2 LDS.32 of the packed scale words per half per block); colsum(B) per token
        // (2 tokens per thread) from smem (kernel: produced by the activation quant / producer)
        const uint8_t* packed_rows0 = smem_p;
        const int r0 = lane >> 2;
        const int wrow = (warp & 3) * 16;
        auto issue = [&](auto tconst, int blk) {
            constexpr int t = decltype(tconst)::value;
            constexpr int par = t & 1;
            const uint64_t db = blk ? desc_b1 : desc_b0;
            wg_fence();
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    wgmma_ss8_u8(acc_blk[par][h], desc_a[h][t] + (uint64_t)((k * 32) >> 4), db + (uint64_t)((k * 32) >> 4), k != 0);
            wg_commit();
        };
        auto promote = [&](auto tconst, int blk) {   // fold the retired block (tile t) into acc_main
            constexpr int par = decltype(tconst)::value & 1;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row0 = ((wg - 1) * 2 + h) * 64 + wrow + r0;
                const uint32_t sw0 = *reinterpret_cast<const uint32_t*>(packed_rows0 + row0 * 80 + 64 + blk * 20480);
                const uint32_t sw1 = *reinterpret_cast<const uint32_t*>(packed_rows0 + (row0 + 8) * 80 + 64 + blk * 20480);
                const int s2_0 = sw0 & 0xff, s2_1 = sw1 & 0xff;
                const int zs_0 = -(int)((sw0 >> 8) & 0xff) * s2_0, zs_1 = -(int)((sw1 >> 8) & 0xff) * s2_1;
                const int2 cs = *reinterpret_cast<const int2*>(smem_b + 2048 - 64 + blk * 32 + (lane & 3) * 8);  // colsum(B) for tokens 2*(lane&3), +1
                acc_main[h][0] += s2_0 * (int)acc_blk[par][h][0] + zs_0 * cs.x;
                acc_main[h][1] += s2_0 * (int)acc_blk[par][h][1] + zs_0 * cs.y;
                acc_main[h][2] += s2_1 * (int)acc_blk[par][h][2] + zs_1 * cs.x;
                acc_main[h][3] += s2_1 * (int)acc_blk[par][h][3] + zs_1 * cs.y;
            }
        };
        using T0 = std::integral_constant<int, 0>;
        using T1 = std::integral_constant<int, 1>;
        using T2 = std::integral_constant<int, 2>;
        // block j uses tile j%3 and activation block j&1; writers run two blocks ahead
        if (writer) { write_tile(0, 0); nbar_arrive(1, 384); write_tile(1, 0); nbar_arrive(2, 384); }
        #pragma unroll 1
        for (int it = 0; it < iters / 3; ++it) {
            // 6 blocks: tiles 0 1 2 0 1 2, activation blocks 0 1 0 1 0 1
            #define SS8U_MATH(TC, TPREV, BLK) \
                STAMP(0, nbar_sync(1 + decltype(TC)::value, 384)); \
                STAMP(1, issue(TC, BLK)); \
                STAMP(2, (wg_wait<1>(), promote(TPREV, BLK ^ 1))); \
                STAMP(3, nbar_arrive(4 + decltype(TPREV)::value, 384));
            #define SS8U_WRITE(TC, ST, NEEDE) \
                if (NEEDE) STAMP(4, nbar_sync(4 + decltype(TC)::value, 384)); \
                STAMP(5, write_tile(decltype(TC)::value, ST)); \
                STAMP(7, nbar_arrive(1 + decltype(TC)::value, 384));
            if (math) {
                SS8U_MATH(T0{}, T2{}, 0) SS8U_MATH(T1{}, T0{}, 1) SS8U_MATH(T2{}, T1{}, 0)
                SS8U_MATH(T0{}, T2{}, 1) SS8U_MATH(T1{}, T0{}, 0) SS8U_MATH(T2{}, T1{}, 1)
            } else if (writer) {
                // writes tiles for blocks j+2: 2 0 1 2 0 1 (first tile 2 of the run needs no E wait)
                SS8U_WRITE(T2{}, 0, (it > 0)) SS8U_WRITE(T0{}, 1, true) SS8U_WRITE(T1{}, 0, true)
                SS8U_WRITE(T2{}, 1, true) SS8U_WRITE(T0{}, 0, true) SS8U_WRITE(T1{}, 1, true)
            }
        }
        // drain: the writers published 2 tiles beyond the math loop; math consumes F, writers consume E
        if (math) { nbar_sync(1 + 0, 384); nbar_sync(1 + 1, 384); }
        if (writer) { nbar_sync(4 + 2, 384); nbar_sync(4 + 0, 384); nbar_sync(4 + 1, 384); }
        if (math) {
            wg_wait<0>();
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc_sink += (int)acc_main[h][j] + (int)acc_blk[0][h][j] + (int)acc_blk[1][h][j];
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
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) out_ph[(blockIdx.x * 12 + warp) * 8 + i] = ph[i];
    }
    sink[blockIdx.x * 384 + threadIdx.x] = acc_sink;  // keep accumulators / decode alive
}

typedef void (*kern_t)(unsigned long long*, unsigned long long*, int*, int, int);

template <int MODE, int FLAGS>
kern_t pick_nh(int nwg, int halves) {
    if (nwg == 2 && halves == 2) return bench_kernel<MODE, 2, 2, FLAGS>;
    if constexpr (MODE != SS8P && MODE != SS8U) {
        if (nwg == 1 && halves == 1) return bench_kernel<MODE, 1, 1, FLAGS>;
        if (nwg == 1 && halves == 2) return bench_kernel<MODE, 1, 2, FLAGS>;
        if (nwg == 1 && halves == 4) return bench_kernel<MODE, 1, 4, FLAGS>;
        if (nwg == 2 && halves == 1) return bench_kernel<MODE, 2, 1, FLAGS>;
        if (nwg == 3 && halves == 1) return bench_kernel<MODE, 3, 1, FLAGS>;
        if (nwg == 3 && halves == 2) return bench_kernel<MODE, 3, 2, FLAGS>;
    }
    return nullptr;
}
template <int MODE>
kern_t pick_f(int flags, int nwg, int halves) {
    // instantiated FLAGS: 0, 1, 8, 9 (decode modes), 2, 4, 6 (ss8d)
    switch (flags) {
        case 0: return pick_nh<MODE, 0>(nwg, halves);
        case 1: return pick_nh<MODE, 1>(nwg, halves);
        case 8: return pick_nh<MODE, 8>(nwg, halves);
        case 9: return pick_nh<MODE, 9>(nwg, halves);
        case 2: if constexpr (MODE == SS8D || MODE == SS8P || MODE == SS8U) return pick_nh<MODE, 2>(nwg, halves); break;
        case 4: if constexpr (MODE == SS8D) return pick_nh<MODE, 4>(nwg, halves); break;
        case 6: if constexpr (MODE == SS8D) return pick_nh<MODE, 6>(nwg, halves); break;
        case 16: if constexpr (MODE == SS8P) return pick_nh<MODE, 16>(nwg, halves); break;
        case 18: if constexpr (MODE == SS8P) return pick_nh<MODE, 18>(nwg, halves); break;
    }
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <mode> <nwg 1..3> <halves 1..4> [iters]\n", argv[0]); return 1; }
    const char* mode_s = argv[1];
    const int nwg = atoi(argv[2]);
    const int halves = atoi(argv[3]);
    int iters = argc > 4 ? atoi(argv[4]) : 1200;
    iters -= iters % 3;
    int mode = -1;
    for (int i = 0; i < NUM_MODES; ++i) if (!strcmp(mode_s, kModeNames[i])) mode = i;
    if (mode < 0) { printf("bad mode\n"); return 1; }
    const int dec_rep = getenv("DEC") ? atoi(getenv("DEC")) : 1;
    const int flags = getenv("FLAGS") ? atoi(getenv("FLAGS")) : 0;
    const bool decode_mode = (mode == RS8D || mode == SS8D || mode == DEC || mode == RS8A || mode == RS8T || mode == SS8P || mode == SS8U);
    const int f = decode_mode ? flags : (flags & 8);

    kern_t k = nullptr;
    switch (mode) {
        case RS8: k = pick_f<RS8>(f, nwg, halves); break;
        case SS8: k = pick_f<SS8>(f, nwg, halves); break;
        case RS16: k = pick_f<RS16>(f, nwg, halves); break;
        case SS16: k = pick_f<SS16>(f, nwg, halves); break;
        case IMMA: k = pick_f<IMMA>(f, nwg, halves); break;
        case IMMAL: k = pick_f<IMMAL>(f, nwg, halves); break;
        case RS8FP8: k = pick_f<RS8FP8>(f, nwg, halves); break;
        case RS8K: k = pick_f<RS8K>(f, nwg, halves); break;
        case RS8D: k = pick_f<RS8D>(f, nwg, halves); break;
        case SS8D: k = pick_f<SS8D>(f, nwg, halves); break;
        case DEC: k = pick_f<DEC>(f, nwg, halves); break;
        case RS8A: k = pick_f<RS8A>(f, nwg, halves); break;
        case RS8T: k = pick_f<RS8T>(f, nwg, halves); break;
        case SS8P: k = pick_f<SS8P>(f, nwg, halves); break;
        case SS8U: k = pick_f<SS8U>(f, nwg, halves); break;
    }
    if (!k) { printf("unsupported mode/flags/nwg/halves combo\n"); return 1; }

    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    const int nsm = prop.multiProcessorCount;
    const size_t smem = kSmemB + kSmemA + kSmemP + 1024;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k));

    unsigned long long* d_out; unsigned long long* d_ph; int* d_sink;
    CK(cudaMalloc(&d_out, nsm * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_ph, nsm * 12 * 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_sink, nsm * 384 * sizeof(int)));
    k<<<nsm, 384, smem>>>(d_out, d_ph, d_sink, 30, dec_rep);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<double> per_run;
    for (int r = 0; r < 5; ++r) {
        k<<<nsm, 384, smem>>>(d_out, d_ph, d_sink, iters, dec_rep);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> h(nsm);
        CK(cudaMemcpy(h.data(), d_out, nsm * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double s = 0; for (auto v : h) s += (double)v;
        per_run.push_back(s / nsm / iters);
    }
    std::sort(per_run.begin(), per_run.end());
    const double ns_stage = per_run[2];  // median over 5 runs, mean over SMs
    const int n_tok = (mode == RS16 || mode == SS16) ? 16 : 8;
    const double rows = (mode == IMMA || mode == IMMAL) ? 256.0 : 64.0 * nwg * halves;
    const double macs = rows * n_tok * 256.0;
    const double ns_norm = ns_stage * (256.0 * 8.0 * 256.0) / macs;
    const double clk = ns_stage * 1.83;
    printf("mode=%-6s dec=%d flags=%d nwg=%d halves=%d regs=%d rows=%.0f ntok=%d : %7.1f ns/stage (%5.0f clk) ; %6.1f MAC/clk/SM ; norm(256x8xK256)=%7.1f ns\n",
           mode_s, dec_rep, flags, nwg, halves, fa.numRegs, rows, n_tok, ns_stage, clk, macs / clk, ns_norm);
    if (mode == RS8D || mode == SS8P || mode == SS8U) {
        std::vector<unsigned long long> hp(nsm * 12 * 8);
        CK(cudaMemcpy(hp.data(), d_ph, hp.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        const int warps[2] = {0, 4};
        const char* lbl[2] = {"writer warp0", "math   warp4"};
        for (int w = 0; w < 2; ++w) {
            printf("   %s clk/stage:", lbl[w]);
            for (int i = 0; i < 8; ++i) {
                double s = 0; for (int b = 0; b < nsm; ++b) s += (double)hp[(b * 12 + warps[w]) * 8 + i];
                printf(" ph%d=%6.0f", i, s / nsm / iters);
            }
            printf("\n");
        }
    }
    return 0;
}
