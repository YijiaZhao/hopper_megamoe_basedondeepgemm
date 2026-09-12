// TMA 2D-box vs 1D bulk-copy microbench for 20480 B pipeline stages (SM90a).
// nvcc -arch=sm_90a -O3 -lcuda -o tma_microbench tma_microbench.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s line %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CKD(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s=0; cuGetErrorString(r,&s); printf("CU err %s line %d: %s\n",#x,__LINE__,s?s:"?"); exit(1);} }while(0)

constexpr int STAGE_BYTES = 20480;
constexpr int STAGES_PER_TASK = 24;
constexpr int TASKS = 8;                 // tasks per CTA (amortize launch overhead)
constexpr int MAX_CTAS = 78;
constexpr int ROWS = 256;
constexpr int STRIDED_ROW_BYTES = 1920;  // 24 * 80
constexpr int BOX_BYTES = 80;
constexpr int MAX_DEPTH = 7;
constexpr size_t SMEM_BYTES = (size_t)MAX_DEPTH * STAGE_BYTES + 128; // always max -> 1 CTA/SM
// 78*8*24*20480 == 78*8*256*1920 : same buffer serves both layouts
constexpr size_t BUF_BYTES = (size_t)MAX_CTAS * TASKS * STAGES_PER_TASK * STAGE_BYTES;
constexpr size_t FLUSH_BYTES = 256ull << 20;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n .reg .pred P1;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @P1 bra DONE;\n bra LAB_WAIT;\n DONE:\n}" :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const void* tmap, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(tmap), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void bulk1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// VARIANT 0: 2D box 80B x 256 rows, row stride 1920 (current kernel)
// VARIANT 1: 2D box 80B x 256 rows, row stride 80 (dense rows, same issue count)
// VARIANT 2: 1D cp.async.bulk 20480 B contiguous
template <int VARIANT>
__global__ void __launch_bounds__(128, 1) bench(const __grid_constant__ CUtensorMap tmap, const uint8_t* __restrict__ buf,
                                                int depth, unsigned long long* timers) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint64_t* bars = (uint64_t*)(smem + MAX_DEPTH * STAGE_BYTES);
  if (threadIdx.x != 0) return;  // single elected thread issues + waits
  for (int i = 0; i < depth; i++) mbar_init(smem_u32(&bars[i]), 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (VARIANT != 2) asm volatile("prefetch.tensormap [%0];" :: "l"(&tmap) : "memory");
  const int cta = blockIdx.x;
  const int total = TASKS * STAGES_PER_TASK;
  auto issue = [&](int g) {
    int slot = g % depth, task = g / STAGES_PER_TASK, s = g % STAGES_PER_TASK;
    uint32_t dst = smem_u32(smem + slot * STAGE_BYTES), bar = smem_u32(&bars[slot]);
    mbar_expect_tx(bar, STAGE_BYTES);
    if (VARIANT == 0) tma2d(dst, &tmap, bar, s * BOX_BYTES, (cta * TASKS + task) * ROWS);
    else if (VARIANT == 1) tma2d(dst, &tmap, bar, 0, ((cta * TASKS + task) * STAGES_PER_TASK + s) * ROWS);
    else bulk1d(dst, buf + ((size_t)(cta * TASKS + task) * STAGES_PER_TASK + s) * STAGE_BYTES, STAGE_BYTES, bar);
  };
  unsigned long long t0 = gtimer();
  for (int g = 0; g < depth; g++) issue(g);
  for (int g = 0; g < total; g++) {
    mbar_wait(smem_u32(&bars[g % depth]), (g / depth) & 1);
    if (g + depth < total) issue(g + depth);
  }
  unsigned long long t1 = gtimer();
  timers[2 * cta] = t0; timers[2 * cta + 1] = t1;
}

static CUtensorMap make_map(void* base, uint64_t inner_bytes, uint64_t rows, uint64_t row_stride) {
  CUtensorMap m;
  cuuint64_t dims[2] = {inner_bytes, rows};
  cuuint64_t strides[1] = {row_stride};
  cuuint32_t box[2] = {BOX_BYTES, ROWS};
  cuuint32_t estr[2] = {1, 1};
  CKD(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, estr,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

typedef void (*kfn_t)(CUtensorMap, const uint8_t*, int, unsigned long long*);

int main() {
  CK(cudaFree(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  printf("device=%s SMs=%d L2=%d MB maxClk=%d MHz\n", prop.name, prop.multiProcessorCount, (int)(prop.l2CacheSize >> 20), clk / 1000);
  uint8_t* buf; CK(cudaMalloc(&buf, BUF_BYTES)); CK(cudaMemset(buf, 1, BUF_BYTES));
  uint8_t* flush; CK(cudaMalloc(&flush, FLUSH_BYTES));
  unsigned long long* timers; CK(cudaMalloc(&timers, 2 * MAX_CTAS * sizeof(unsigned long long)));
  std::vector<unsigned long long> h_t(2 * MAX_CTAS);
  CUtensorMap map_strided = make_map(buf, STRIDED_ROW_BYTES, (uint64_t)MAX_CTAS * TASKS * ROWS, STRIDED_ROW_BYTES);
  CUtensorMap map_dense = make_map(buf, BOX_BYTES, (uint64_t)MAX_CTAS * TASKS * STAGES_PER_TASK * ROWS, BOX_BYTES);
  kfn_t fns[3] = {bench<0>, bench<1>, bench<2>};
  const char* names[3] = {"2D 80Bx256 stride1920", "2D 80Bx256 stride80  ", "1D bulk 20480B       "};
  for (int v = 0; v < 3; v++) CK(cudaFuncSetAttribute((const void*)fns[v], cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const int ctas_list[2] = {20, 78}, depth_list[3] = {1, 4, 7}, REPS = 50;
  printf("stages/CTA=%d (%d tasks x %d), stage=%d B, buffer=%.0f MB, L2 flush=256 MiB memset per rep, median of %d\n",
         TASKS * STAGES_PER_TASK, TASKS, STAGES_PER_TASK, STAGE_BYTES, BUF_BYTES / 1e6, REPS);
  printf("%-22s %4s %5s | %10s %12s %12s | %9s %9s\n", "variant", "CTAs", "depth", "kernel_us", "stage_us_ev", "stage_us_gt", "GB/s/SM", "GB/s_agg");
  for (int ci = 0; ci < 2; ci++) for (int di = 0; di < 3; di++) for (int v = 0; v < 3; v++) {
    int ctas = ctas_list[ci], depth = depth_list[di];
    const CUtensorMap& m = (v == 0) ? map_strided : map_dense;
    std::vector<float> ev; std::vector<double> gt;
    for (int r = 0; r < REPS + 2; r++) {
      CK(cudaMemsetAsync(flush, r & 0xff, FLUSH_BYTES));
      CK(cudaEventRecord(e0));
      fns[v]<<<ctas, 128, SMEM_BYTES>>>(m, buf, depth, timers);
      CK(cudaEventRecord(e1));
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      CK(cudaMemcpy(h_t.data(), timers, 2 * ctas * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      unsigned long long mn = ~0ull, mx = 0;
      for (int c = 0; c < ctas; c++) { mn = std::min(mn, h_t[2 * c]); mx = std::max(mx, h_t[2 * c + 1]); }
      if (r >= 2) { ev.push_back(ms * 1000.f); gt.push_back((mx - mn) / 1000.0); }
    }
    std::sort(ev.begin(), ev.end()); std::sort(gt.begin(), gt.end());
    double kus = ev[REPS / 2], gus = gt[REPS / 2];
    double stage_ev = kus / (TASKS * STAGES_PER_TASK), stage_gt = gus / (TASKS * STAGES_PER_TASK);
    double gbs_sm = STAGE_BYTES / (stage_gt * 1e3);  // bytes / ns = GB/s
    printf("%-22s %4d %5d | %10.2f %12.4f %12.4f | %9.1f %9.1f\n", names[v], ctas, depth, kus, stage_ev, stage_gt, gbs_sm, gbs_sm * ctas);
  }
  return 0;
}
