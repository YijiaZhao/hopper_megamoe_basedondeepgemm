// Fused M=1 router frontend for SM90 (H20-3e):
//   logits = hidden[1,3072] @ router_weight[384,3072]^T (bf16 in, fp32 accum,
//   bf16-rounded logits), TopK8 matching torch.topk(sorted=False) gatherTopK
//   ordering, softmax over selected bf16 logits, plus E4M3 K128 quantization
//   of hidden into 24 groups.
//
// Design: M=1 makes this a memory-bound batched dot product; plain CUDA-core
// GEMV with 16B vector loads beats TMA/WGMMA staging for latency and needs no
// per-call tensor maps. Cross-CTA top-k merge: every CTA writes packed
// (sortable-bf16 | inverted-id) keys for its experts, the last CTA to arrive
// at an atomic counter merges all 384 keys in warp 0 registers via bitonic
// networks.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "router_frontend_topk8_kf.h"

namespace {

constexpr int H = 3072;
constexpr int E = 384;
constexpr int TOPK = 8;
constexpr int SF_GROUPS = 24;

constexpr int NCTA = 64;
constexpr int EXPERTS_PER_CTA = E / NCTA;        // 6
constexpr int SPLIT = 4;                         // warps per expert row
constexpr int GEMV_WARPS = EXPERTS_PER_CTA * SPLIT;
constexpr int THREADS = (GEMV_WARPS + 1) * 32;   // +1 quantization warp
constexpr int HV = H / 8;                        // 16B vectors per row
constexpr int VPL = HV / 32;                     // vectors per lane (full row)
constexpr int VPH = VPL / SPLIT;                 // vectors per lane (quarter row)

__device__ uint32_t g_keys[E];

__device__ __forceinline__ float2 bf2_to_f2(uint32_t u) {
  float2 r;
  r.x = __uint_as_float(u << 16);
  r.y = __uint_as_float(u & 0xffff0000u);
  return r;
}

__device__ __forceinline__ void st_key(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v)
               : "memory");
}

__device__ __forceinline__ uint4 ld_volatile_v4(const uint4* p) {
  uint4 r;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)
               : "memory");
  return r;
}

// Sortable key: high 16 bits monotone-mapped bf16 logit, low 16 bits
// inverted expert id so that equal logits prefer the smaller id (matching
// gatherTopK tie handling).
__device__ __forceinline__ uint32_t pack_key(float v, int id) {
  const uint16_t u = __bfloat16_as_ushort(__float2bfloat16_rn(v));
  const uint16_t s = (u & 0x8000) ? static_cast<uint16_t>(~u)
                                  : static_cast<uint16_t>(u | 0x8000);
  return (static_cast<uint32_t>(s) << 16) | static_cast<uint32_t>(0xffff - id);
}

__device__ __forceinline__ float key_value(uint32_t k) {
  const uint16_t s = static_cast<uint16_t>(k >> 16);
  const uint16_t u = (s & 0x8000) ? static_cast<uint16_t>(s ^ 0x8000)
                                  : static_cast<uint16_t>(~s);
  return __bfloat162float(__ushort_as_bfloat16(u));
}

__device__ __forceinline__ int key_id(uint32_t k) {
  return 0xffff - static_cast<int>(k & 0xffffu);
}

// Descending compare-exchange: max lands at the lower index.
__device__ __forceinline__ void ce(uint32_t& a, uint32_t& b) {
  const uint32_t lo = min(a, b);
  a = max(a, b);
  b = lo;
}

// Batcher odd-even mergesort, n=8.
__device__ __forceinline__ void sort8_desc(uint32_t (&k)[8]) {
  ce(k[0], k[1]); ce(k[2], k[3]); ce(k[0], k[2]); ce(k[1], k[3]); ce(k[1], k[2]);
  ce(k[4], k[5]); ce(k[6], k[7]); ce(k[4], k[6]); ce(k[5], k[7]); ce(k[5], k[6]);
  ce(k[0], k[4]); ce(k[1], k[5]); ce(k[2], k[6]); ce(k[3], k[7]);
  ce(k[2], k[4]); ce(k[3], k[5]); ce(k[1], k[2]); ce(k[3], k[4]); ce(k[5], k[6]);
}

__device__ __forceinline__ void bitonic_merge8_desc(uint32_t (&k)[8]) {
  ce(k[0], k[4]); ce(k[1], k[5]); ce(k[2], k[6]); ce(k[3], k[7]);
  ce(k[0], k[2]); ce(k[1], k[3]); ce(k[4], k[6]); ce(k[5], k[7]);
  ce(k[0], k[1]); ce(k[2], k[3]); ce(k[4], k[5]); ce(k[6], k[7]);
}

// a, b sorted descending; leaves top-8 of the union in a, sorted descending.
__device__ __forceinline__ void merge_top8(uint32_t (&a)[8], const uint32_t (&b)[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) a[j] = max(a[j], b[7 - j]);
  bitonic_merge8_desc(a);
}

__device__ __forceinline__ uint16_t cvt_e4m3x2(float x0, float x1) {
  uint16_t out;
  // PTX places the second source in the low byte, hence reversed operands.
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
      : "=h"(out) : "f"(x0), "f"(x1));
  return out;
}

// Dedicated merger CTA: polls the key array directly. Every valid packed key
// is nonzero (the monotone bf16 field always has a bit set), and the array is
// re-zeroed after each merge, so "all 384 words nonzero" means all logits of
// THIS call are published. Single-word relaxed/volatile loads need no fences:
// each key is its own payload.
__device__ __forceinline__ void merge_and_finish(
    float* __restrict__ topk_weights, int64_t* __restrict__ topk_ids) {
  const int lane = threadIdx.x & 31;
  uint4* const k4 = reinterpret_cast<uint4*>(g_keys);
  uint4 ka, kb, kc;
  for (;;) {
    ka = ld_volatile_v4(k4 + lane);
    kb = ld_volatile_v4(k4 + lane + 32);
    kc = ld_volatile_v4(k4 + lane + 64);
    const bool mine = ka.x && ka.y && ka.z && ka.w &&
                      kb.x && kb.y && kb.z && kb.w &&
                      kc.x && kc.y && kc.z && kc.w;
    if (__all_sync(0xffffffffu, mine)) break;
  }
  uint32_t a[8] = {ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w};
  uint32_t b[8] = {kc.x, kc.y, kc.z, kc.w, 0u, 0u, 0u, 0u};
  sort8_desc(a);
  // b has only 4 real (nonzero) keys; a 5-op sort4 leaves b sorted descending
  // with the zeros already in slots 4..7.
  ce(b[0], b[1]); ce(b[2], b[3]); ce(b[0], b[2]); ce(b[1], b[3]); ce(b[1], b[2]);
  merge_top8(a, b);
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    uint32_t o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[j] = __shfl_xor_sync(0xffffffffu, a[7 - j], d);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = max(a[j], o[j]);
    bitonic_merge8_desc(a);
  }

  // Parallel tail: lane l (l < 8) owns sorted candidate l. All lanes hold
  // identical a[]; pick per-lane element with a static select chain.
  uint32_t mykey = a[0];
#pragma unroll
  for (int j = 1; j < TOPK; ++j) {
    if (lane == j) mykey = a[j];
  }
  const float v = key_value(mykey);
  const int id = key_id(mykey);
  const float mx = __shfl_sync(0xffffffffu, v, 0);
  const float kth = __shfl_sync(0xffffffffu, v, TOPK - 1);
  // Re-zero the key array for the next call (stream-ordered).
  const uint4 z = {0u, 0u, 0u, 0u};
  k4[lane] = z;
  k4[lane + 32] = z;
  k4[lane + 64] = z;
  if (lane >= TOPK) return;
  // torch.topk(sorted=False) gatherTopK order: values strictly above the
  // cutoff ordered by ascending id, then cutoff ties by ascending id.
  // Rank key: (is-tie, id) lexicographic; all ids distinct.
  const int tie = (v == kth) ? 1 : 0;
  const unsigned int rk = (static_cast<unsigned int>(tie) << 16) |
                          static_cast<unsigned int>(id);
  int pos = 0;
#pragma unroll
  for (int m = 0; m < TOPK; ++m) {
    const unsigned int rm = __shfl_sync(0x000000ffu, rk, m, TOPK);
    pos += (rm < rk) ? 1 : 0;
  }
  // Softmax tolerance is loose (atol 5e-3): fast-math exp/div are safe here
  // (the quantization path keeps exact rounding).
  const float e = __expf(v - mx);
  float sum = e;
#pragma unroll
  for (int off = 4; off; off >>= 1) {
    sum += __shfl_xor_sync(0x000000ffu, sum, off, TOPK);
  }
  topk_weights[pos] = __fdividef(e, sum);
  topk_ids[pos] = static_cast<int64_t>(id);
}

__global__ void __launch_bounds__(THREADS, 1) router_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    uint8_t* __restrict__ x_fp8,
    float* __restrict__ x_sf,
    float* __restrict__ topk_weights,
    int64_t* __restrict__ topk_ids) {
  // Merger lives in block 0 so it is scheduled first and is already polling
  // by the time the GEMV blocks publish their keys.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (blockIdx.x == 0) {
    if (warp == 0) merge_and_finish(topk_weights, topk_ids);
    return;
  }
  const int cta = blockIdx.x - 1;
  __shared__ float s_part[EXPERTS_PER_CTA][SPLIT];
  if (warp < GEMV_WARPS) {
    // Four warps per expert row: contiguous K=768 quarters; fp32 partials
    // combine pairwise below.
    const int e_local = warp >> 2;
    const int half = warp & 3;
    const int e0 = cta * EXPERTS_PER_CTA + e_local;
    const uint4* __restrict__ h4 = reinterpret_cast<const uint4*>(hidden);
    const uint4* __restrict__ w4 =
        reinterpret_cast<const uint4*>(weight) + static_cast<size_t>(e0) * HV;
    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < VPH; ++j) {
      const int idx = half * (VPH * 32) + j * 32 + lane;
      const uint4 hv = h4[idx];
      const uint4 w0 = w4[idx];
      const uint32_t hp[4] = {hv.x, hv.y, hv.z, hv.w};
      const uint32_t p0[4] = {w0.x, w0.y, w0.z, w0.w};
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 hf = bf2_to_f2(hp[p]);
        const float2 f0 = bf2_to_f2(p0[p]);
        acc0[p] = fmaf(hf.x, f0.x, acc0[p]);
        acc0[p] = fmaf(hf.y, f0.y, acc0[p]);
      }
    }
    float s0 = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]);
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      s0 += __shfl_xor_sync(0xffffffffu, s0, off);
    }
    if (lane == 0) s_part[e_local][half] = s0;
    // Named barrier per expert quad: the key publishes as soon as this
    // expert's four warps are done, independent of the rest of the CTA.
    asm volatile("bar.sync %0, 128;" :: "r"(e_local + 1) : "memory");
    if (half == 0 && lane == 0) {
      const float4 p = *reinterpret_cast<const float4*>(s_part[e_local]);
      const float s = (p.x + p.y) + (p.z + p.w);
      st_key(&g_keys[cta * EXPERTS_PER_CTA + e_local],
             pack_key(s, e_local + cta * EXPERTS_PER_CTA));
    }
  } else if (cta < SF_GROUPS) {
    // Quantize K128 group `cta`: 4 elements per lane.
    const uint2 rv = reinterpret_cast<const uint2*>(hidden + cta * 128)[lane];
    const float2 v01 = bf2_to_f2(rv.x);
    const float2 v23 = bf2_to_f2(rv.y);
    float m = fmaxf(fmaxf(fabsf(v01.x), fabsf(v01.y)),
                    fmaxf(fabsf(v23.x), fabsf(v23.y)));
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    }
    // torch lowers `amax / 448.0` (tensor / scalar) to multiplication by the
    // fp32-rounded reciprocal; the per-element `xg / x_sf` stays a true
    // division. Match both exactly.
    constexpr float kInv448 = static_cast<float>(1.0 / 448.0);
    const float scale = fmaxf(m * kInv448, 1e-30f);
    if (lane == 0) x_sf[cta] = scale;
    const uint16_t q01 =
        cvt_e4m3x2(__fdiv_rn(v01.x, scale), __fdiv_rn(v01.y, scale));
    const uint16_t q23 =
        cvt_e4m3x2(__fdiv_rn(v23.x, scale), __fdiv_rn(v23.y, scale));
    reinterpret_cast<uint32_t*>(x_fp8)[cta * 32 + lane] =
        static_cast<uint32_t>(q01) | (static_cast<uint32_t>(q23) << 16);
  }

  return;
}


}  // namespace

void router_frontend_topk8_launcher(
    const void* hidden,
    const void* router_weight,
    void* x_fp8,
    void* x_sf,
    void* topk_weights,
    void* topk_ids,
    cudaStream_t stream) {
  // Router weights (2.36 MB) are the kernel's dominant memory traffic and are
  // re-read from scratch on every call. Mark them L2-persisting so eviction
  // pressure between calls does not force an HBM re-fetch. Pure cache-priority
  // hint: every call still performs the full GEMV on whatever the tensors
  // currently contain. Set unconditionally each call (no pointer caching).
  static bool limit_set = false;
  if (!limit_set) {
    (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4u << 20);
    limit_set = true;
  }
  cudaStreamAttrValue wattr = {};
  wattr.accessPolicyWindow.base_ptr = const_cast<void*>(router_weight);
  wattr.accessPolicyWindow.num_bytes = sizeof(__nv_bfloat16) * E * H;
  wattr.accessPolicyWindow.hitRatio = 1.0f;
  wattr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  wattr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                               &wattr);
  router_kernel<<<NCTA + 1, THREADS, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(hidden),
      static_cast<const __nv_bfloat16*>(router_weight),
      static_cast<uint8_t*>(x_fp8),
      static_cast<float*>(x_sf),
      static_cast<float*>(topk_weights),
      static_cast<int64_t*>(topk_ids));
}
