#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>

#include "qoq_router_frontend_kf.h"

namespace {

namespace cg = cooperative_groups;
constexpr int H = 4096;
constexpr int E = 128;
constexpr int BM = 64;
constexpr int WIDTH = 64;
constexpr int TOPK = 6;
constexpr int CLUSTER_CTAS = 8;
constexpr int STAGES = 2;

struct alignas(1024) PipelineStage {
  __nv_bfloat16 a[8][BM][WIDTH];
  __nv_bfloat16 b[8][16][WIDTH];
};

struct ResultStorage {
  float logits[BM][16];
  float candidate_value[BM][TOPK];
  int candidate_id[BM][TOPK];
  float selected[BM][TOPK];
};

union alignas(1024) SharedStorage {
  PipelineStage stage[STAGES];
  ResultStorage result;
};

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init_one(uint64_t* bar) {
  const uint32_t addr = smem_u32(bar);
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], 1;\n"
      :: "r"(addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect(
    uint64_t* bar, uint32_t bytes) {
  const uint32_t addr = smem_u32(bar);
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
      :: "r"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbarrier_ready(
    uint64_t* bar, uint32_t old_phase) {
  uint32_t ready;
  const uint32_t addr = smem_u32(bar);
  asm volatile(
      "{ .reg .pred p; "
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
      "selp.u32 %0, 1, 0, p; }\n"
      : "=r"(ready) : "r"(addr), "r"(old_phase) : "memory");
  return ready != 0;
}

__device__ __forceinline__ void tma_load_2d(
    void* dst,
    const CUtensorMap* map,
    uint64_t* bar,
    int32_t coord0,
    int32_t coord1) {
  const uint32_t dst_addr = smem_u32(dst);
  const uint32_t bar_addr = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(dst_addr), "l"(map), "r"(coord0), "r"(coord1),
         "r"(bar_addr) : "memory");
}

__device__ __forceinline__ void issue_stage_load(
    PipelineStage& stage,
    const CUtensorMap* hidden_map,
    const CUtensorMap* weight_map,
    uint64_t* full,
    int group,
    int expert_base) {
  mbarrier_arrive_expect(full, sizeof(PipelineStage));
#pragma unroll
  for (int half = 0; half < 8; ++half) {
    const int k = group * 512 + half * WIDTH;
    tma_load_2d(&stage.a[half][0][0], hidden_map, full, k, 0);
    tma_load_2d(&stage.b[half][0][0], weight_map, full, k, expert_base);
  }
}

__device__ __forceinline__ uint64_t encode_desc(uint64_t x) {
  return (x & 0x3ffffULL) >> 4;
}

__device__ __forceinline__ uint64_t make_kmajor_desc(const void* ptr) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = encode_desc(addr);
  desc |= 1ULL << 16;
  desc |= encode_desc(1024) << 32;
  desc |= 1ULL << 62;
  return desc;
}

__device__ __forceinline__ float neg_inf() {
  return __int_as_float(static_cast<int>(0xff800000u));
}

__device__ __forceinline__ float round_to_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n16k16(
    float (&d)[8], const void* a, const void* b) {
  const uint64_t desc_a = make_kmajor_desc(a);
  const uint64_t desc_b = make_kmajor_desc(b);
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,1,1,1,0,0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void insert_top6(
    float value, int id, float (&best_value)[TOPK], int (&best_id)[TOPK]) {
  if (value > best_value[TOPK - 1]) {
    int pos = TOPK - 1;
    while (pos > 0 && value > best_value[pos - 1]) {
      best_value[pos] = best_value[pos - 1];
      best_id[pos] = best_id[pos - 1];
      --pos;
    }
    best_value[pos] = value;
    best_id[pos] = id;
  }
}

__device__ __forceinline__ void partial_amax_stage(
    const PipelineStage& stage,
    int rank,
    int m,
    float* __restrict__ scratch) {
  const int lane = threadIdx.x & 31;
  const int worker = (threadIdx.x - 128) >> 5;
  const int workers = (blockDim.x - 128) >> 5;
  for (int row = worker; row < m; row += workers) {
    float local_max = 0.0f;
#pragma unroll
    for (int half = 0; half < 8; ++half) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int kk = lane * 2 + j;
        const int sk = kk ^ ((row & 7) << 3);
        const float value = __bfloat162float(stage.a[half][row][sk]);
        local_max = fmaxf(local_max, fabsf(value));
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_max = fmaxf(
          local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    }
    if (lane == 0) scratch[rank * BM + row] = local_max;
  }
}

__device__ __forceinline__ uint64_t quantize8(
    const __nv_bfloat16* values, float scale) {
  uint64_t packed = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int q = __float2int_rn(__bfloat162float(values[j]) / scale);
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    packed |= static_cast<uint64_t>(
                  static_cast<uint8_t>(static_cast<int8_t>(q)))
              << (8 * j);
  }
  return packed;
}

__device__ __forceinline__ void quantize_row(
    SharedStorage& sm,
    const __nv_bfloat16* __restrict__ hidden,
    int8_t* __restrict__ x_int8,
    float* __restrict__ x_sf,
    int row) {
  float local_max = 0.0f;
  const __nv_bfloat16* row_in = hidden + row * H;
  for (int q = threadIdx.x; q < H / 8; q += blockDim.x) {
    const __nv_bfloat16* values = row_in + q * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      local_max = fmaxf(local_max, fabsf(__bfloat162float(values[j])));
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(
        local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
  }
  float* warp_max = reinterpret_cast<float*>(&sm);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_max[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    float block_max = lane < ((blockDim.x + 31) >> 5) ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_max = fmaxf(
          block_max, __shfl_down_sync(0xffffffffu, block_max, offset));
    }
    if (lane == 0) {
      const float scale = fmaxf(block_max * (1.0f / 127.0f), 1.0e-30f);
      warp_max[0] = scale;
#pragma unroll
      for (int sf = 0; sf < 32; ++sf) x_sf[row * 32 + sf] = scale;
    }
  }
  __syncthreads();
  const float scale = warp_max[0];
  for (int q = threadIdx.x; q < H / 8; q += blockDim.x) {
    const int k = q * 8;
    *reinterpret_cast<uint64_t*>(x_int8 + row * H + k) =
        quantize8(row_in + k, scale);
  }
  __syncthreads();
}

__device__ __forceinline__ void local_top6(
    SharedStorage& sm, int row, int expert_base) {
  float best_value[TOPK];
  int best_id[TOPK];
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    best_value[j] = neg_inf();
    best_id[j] = -1;
  }
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    insert_top6(
        sm.result.logits[row][e], expert_base + e, best_value, best_id);
  }
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    sm.result.candidate_value[row][j] = best_value[j];
    sm.result.candidate_id[row][j] = best_id[j];
  }
}

__device__ __forceinline__ void merge_top6(
    cg::cluster_group cluster,
    SharedStorage& sm,
    int row,
    int64_t* __restrict__ topk_ids) {
  float best_value[TOPK];
  int best_id[TOPK];
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    best_value[j] = neg_inf();
    best_id[j] = -1;
  }
#pragma unroll
  for (int rank = 0; rank < CLUSTER_CTAS; ++rank) {
    const float* remote_value =
        cluster.map_shared_rank(&sm.result.candidate_value[0][0], rank);
    const int* remote_id =
        cluster.map_shared_rank(&sm.result.candidate_id[0][0], rank);
#pragma unroll
    for (int j = 0; j < TOPK; ++j) {
      insert_top6(
          remote_value[row * TOPK + j], remote_id[row * TOPK + j],
          best_value, best_id);
    }
  }

  const float kth = best_value[TOPK - 1];
#pragma unroll
  for (int i = 1; i < TOPK; ++i) {
    const float value = best_value[i];
    const int id = best_id[i];
    const int tie = value == kth;
    int j = i;
    while (j > 0) {
      const int prev_tie = best_value[j - 1] == kth;
      if (prev_tie < tie || (prev_tie == tie && best_id[j - 1] <= id)) break;
      best_value[j] = best_value[j - 1];
      best_id[j] = best_id[j - 1];
      --j;
    }
    best_value[j] = value;
    best_id[j] = id;
  }
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    sm.result.selected[row][j] = best_value[j];
    topk_ids[row * TOPK + j] = static_cast<int64_t>(best_id[j]);
  }
}

__device__ __forceinline__ void softmax_top6(
    SharedStorage& sm,
    int m,
    float* __restrict__ topk_weights) {
  const int tid = threadIdx.x;
  if (tid >= 128) return;
  const int subgroup = tid >> 3;
  const int sublane = tid & 7;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int first_row = pass * 32 + subgroup * 2;
    float elem[2];
    float mx[2];
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const int row = first_row + b;
      elem[b] = (row < m && sublane < TOPK)
                    ? sm.result.selected[row][sublane]
                    : neg_inf();
      mx[b] = elem[b];
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const float other =
            __shfl_xor_sync(0xffffffffu, mx[b], offset, 8);
        mx[b] = mx[b] > other ? mx[b] : other;
      }
    }
    float ex[2];
    float sum[2];
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      ex[b] = expf(elem[b] - mx[b]);
      sum[b] = ex[b];
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        sum[b] += __shfl_xor_sync(0xffffffffu, sum[b], offset, 8);
      }
    }
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const int row = first_row + b;
      if (row < m && sublane < TOPK) {
        topk_weights[row * TOPK + sublane] = ex[b] / sum[b];
      }
    }
  }
}

template <int QWARPS, bool ROW_QUANT>
__global__ __cluster_dims__(8, 1, 1) __launch_bounds__(128 + 32 * QWARPS, 1)
void router_frontend_kernel(
    const __grid_constant__ CUtensorMap hidden_map,
    const __grid_constant__ CUtensorMap weight_map,
    const __nv_bfloat16* __restrict__ hidden,
    int8_t* __restrict__ x_int8,
    float* __restrict__ x_sf,
    float* __restrict__ topk_weights,
    int64_t* __restrict__ topk_ids,
    int m) {
  __shared__ SharedStorage sm;
  __shared__ alignas(8) uint64_t full[STAGES];

  const int tid = threadIdx.x;
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int expert_base = rank * 16;
  float* scratch = reinterpret_cast<float*>(x_int8);

  if constexpr (ROW_QUANT) {
    if (blockIdx.x >= CLUSTER_CTAS) {
      const int row = static_cast<int>(blockIdx.x) - CLUSTER_CTAS;
      if (tid == 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");
      __syncthreads();
      if (row < m) quantize_row(sm, hidden, x_int8, x_sf, row);
      __syncthreads();
      if (tid == 0) {
        asm volatile(
            "fence.release.gpu;\n"
            "griddepcontrol.launch_dependents;\n" ::: "memory");
      }
      return;
    }
  }

  if (tid == 0) {
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      mbarrier_init_one(&full[stage]);
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  }
  __syncthreads();

  if (tid == 128) {
#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      issue_stage_load(
          sm.stage[stage], &hidden_map, &weight_map, &full[stage],
          stage, expert_base);
    }
  }

  float d[4][8];
#pragma unroll
  for (int part = 0; part < 4; ++part) {
#pragma unroll
    for (int i = 0; i < 8; ++i) d[part][i] = 0.0f;
  }

#pragma unroll 1
  for (int group = 0; group < 8; ++group) {
    const int stage = group % STAGES;
    const uint32_t phase = (group / STAGES) & 1;
    while (!mbarrier_ready(&full[stage], phase)) {}

    if (tid < 128) {
      wgmma_fence();
#pragma unroll
      for (int half = 0; half < 8; ++half) {
#pragma unroll
        for (int k = 0; k < WIDTH; k += 16) {
          wgmma_m64n16k16(
              d[group >> 1],
              &sm.stage[stage].a[half][0][k],
              &sm.stage[stage].b[half][0][k]);
        }
      }
      wgmma_commit();
      wgmma_wait();
    } else if constexpr (!ROW_QUANT) {
      if (group == rank) {
        partial_amax_stage(sm.stage[stage], rank, m, scratch);
      }
    }
    __syncthreads();
    if (tid == 128 && group + STAGES < 8) {
      issue_stage_load(
          sm.stage[stage], &hidden_map, &weight_map, &full[stage],
          group + STAGES, expert_base);
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float lo = d[0][i] + d[1][i];
    const float hi = d[2][i] + d[3][i];
    d[0][i] = lo + hi;
  }

  if (tid < 128) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int r0 = warp * 16 + lane / 4;
    const int col = 2 * (lane & 3);
    sm.result.logits[r0][col] = round_to_bf16(d[0][0]);
    sm.result.logits[r0][col + 1] = round_to_bf16(d[0][1]);
    sm.result.logits[r0 + 8][col] = round_to_bf16(d[0][2]);
    sm.result.logits[r0 + 8][col + 1] = round_to_bf16(d[0][3]);
    sm.result.logits[r0][col + 8] = round_to_bf16(d[0][4]);
    sm.result.logits[r0][col + 9] = round_to_bf16(d[0][5]);
    sm.result.logits[r0 + 8][col + 8] = round_to_bf16(d[0][6]);
    sm.result.logits[r0 + 8][col + 9] = round_to_bf16(d[0][7]);
  }
  __syncthreads();

  if (tid < m) local_top6(sm, tid, expert_base);
  cluster.sync();

  if (rank == 0 && tid < m) {
    merge_top6(cluster, sm, tid, topk_ids);
    if constexpr (!ROW_QUANT) {
      float row_max = 0.0f;
#pragma unroll
      for (int r = 0; r < CLUSTER_CTAS; ++r) {
        row_max = fmaxf(row_max, scratch[r * BM + tid]);
      }
      const float scale = fmaxf(row_max * (1.0f / 127.0f), 1.0e-30f);
#pragma unroll
      for (int sf = 0; sf < 32; ++sf) x_sf[tid * 32 + sf] = scale;
    }
  }
  __syncthreads();
  if (rank == 0) softmax_top6(sm, m, topk_weights);
  cluster.sync();

  if constexpr (!ROW_QUANT) {
    for (int q = tid; q < m * 64; q += blockDim.x) {
      const int row = q >> 6;
      const int kk = (q & 63) * 8;
      const __nv_bfloat16* src = hidden + row * H + rank * 512 + kk;
      *reinterpret_cast<uint64_t*>(x_int8 + row * H + rank * 512 + kk) =
          quantize8(src, x_sf[row * 32]);
    }
  }
  __syncthreads();
  if (tid == 0) {
    asm volatile(
        "fence.release.gpu;\n"
        "griddepcontrol.launch_dependents;\n" ::: "memory");
  }
}

void encode_map(
    CUtensorMap* map,
    const void* ptr,
    uint64_t rows,
    uint32_t tile_rows,
    CUtensorMapFloatOOBfill oob_fill) {
  constexpr uint32_t rank = 2;
  const uint64_t dims[rank] = {H, rows};
  const uint64_t strides[rank - 1] = {H * sizeof(__nv_bfloat16)};
  const uint32_t box[rank] = {WIDTH, tile_rows};
  const uint32_t elem_strides[rank] = {1, 1};
  cuTensorMapEncodeTiled(
      map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      const_cast<void*>(ptr),
      dims,
      strides,
      box,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      oob_fill);
}

template <int QWARPS, bool ROW_QUANT>
void launch_router(
    CUtensorMap hidden_map,
    CUtensorMap weight_map,
    const __nv_bfloat16* hidden,
    int8_t* x_int8,
    float* x_sf,
    float* topk_weights,
    int64_t* topk_ids,
    int m,
    cudaStream_t stream) {
  cudaLaunchAttribute attribute[1]{};
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config{};
  const int blocks = ROW_QUANT
      ? CLUSTER_CTAS + ((m + CLUSTER_CTAS - 1) & -CLUSTER_CTAS)
      : CLUSTER_CTAS;
  config.gridDim = dim3(blocks, 1, 1);
  config.blockDim = dim3(128 + 32 * QWARPS, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(
      &config, router_frontend_kernel<QWARPS, ROW_QUANT>,
      hidden_map, weight_map, hidden,
      x_int8, x_sf, topk_weights, topk_ids, m);
}

}  // namespace

void qoq_router_frontend_launcher(
    const void* hidden,
    const void* router_weight,
    void* x_int8,
    void* x_sf,
    void* topk_weights,
    void* topk_ids,
    int m,
    cudaStream_t stream) {
  CUtensorMap hidden_map{};
  CUtensorMap weight_map{};
  encode_map(
      &hidden_map, hidden, static_cast<uint64_t>(m), BM,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  encode_map(
      &weight_map, router_weight, E, 16,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  const auto* hidden_ptr = static_cast<const __nv_bfloat16*>(hidden);
  auto* x_int8_ptr = static_cast<int8_t*>(x_int8);
  auto* x_sf_ptr = static_cast<float*>(x_sf);
  auto* topk_weights_ptr = static_cast<float*>(topk_weights);
  auto* topk_ids_ptr = static_cast<int64_t*>(topk_ids);
  if (m <= 8) {
    launch_router<4, false>(hidden_map, weight_map, hidden_ptr,
        x_int8_ptr, x_sf_ptr, topk_weights_ptr, topk_ids_ptr, m, stream);
  } else if (m <= 16) {
    launch_router<2, true>(hidden_map, weight_map, hidden_ptr,
        x_int8_ptr, x_sf_ptr, topk_weights_ptr, topk_ids_ptr, m, stream);
  } else if (m <= 32) {
    launch_router<4, true>(hidden_map, weight_map, hidden_ptr,
        x_int8_ptr, x_sf_ptr, topk_weights_ptr, topk_ids_ptr, m, stream);
  } else if (m <= 48) {
    launch_router<6, true>(hidden_map, weight_map, hidden_ptr,
        x_int8_ptr, x_sf_ptr, topk_weights_ptr, topk_ids_ptr, m, stream);
  } else {
    launch_router<8, true>(hidden_map, weight_map, hidden_ptr,
        x_int8_ptr, x_sf_ptr, topk_weights_ptr, topk_ids_ptr, m, stream);
  }
}
