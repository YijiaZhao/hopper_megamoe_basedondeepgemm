#!/usr/bin/env bash
# CPU-only compile check of one SM90 fused MegaMoE kernel instantiation (the same
# template arguments the host JIT emits for the BM8 tier), with ptxas' register /
# spill / smem report. Lets a task-shape change be checked and its register
# footprint read while the GPUs are busy.
#   bash scripts/compile_check_fused_kernel.sh <quant mxfp4|qoq> <l1_tiles 1|2> <l2_tiles 1|2> <stages> [outdir]
#   HOT_SPLIT=1|2 sets kHotSplitLevel (DG_FP4_HOT_SPLIT); STREAMK=true compiles the stream-K paths in.
#   (K128 blocks per stage follow the task shape: 2, or 1 with wide tiles.)
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
QUANT=${1:-mxfp4}; L1T=${2:-1}; L2T=${3:-1}; STAGES=${4:-4}
OUT=${5:-/tmp/dg_compile_check/${QUANT}_bn$((256 * L1T))x$((256 * L2T))_s${STAGES}}
mkdir -p "$OUT"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CUTLASS=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
if [ "$QUANT" = qoq ]; then SYM=sm90_qoq_mega_moe_h20_fused_impl; MX=false; QQ=true; QIS2=true; QPF=true
else SYM=sm90_mxfp4_mega_moe_h20_fused_impl; MX=true; QQ=false; QIS2=false; QPF=false; fi
cat > "$OUT/kernel.cu" <<CU
#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
${EXTRA_DEFINES:-}
#define sm90_nvfp4_mega_moe_h200_fused_impl $SYM
#include <deep_gemm/impls/sm90_fp4_mega_moe_h20_fused.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&$SYM<
        /* kNumMaxTokensPerRank */ 2,
        /* kNumExpertsPerWave */ 16,
        /* BLOCK_M */ 8,
        /* BLOCK_N */ 256,
        /* kNumMaxPoolTokens */ 9600,
        /* kNumPaddedSFPoolTokens */ 153600,
        /* kNumStages */ $STAGES,
        /* kActivationClamp */ 0x1.4p+3f,
        /* kFastMath */ true,
        /* kSwapABRequested */ true,
        /* kSingleActiveDispatchWarp */ true,
        /* kUseMode2RowDecoder */ true,
        /* kUseInterleavedScheduler */ true,
        /* kMXFP4 */ $MX,
        /* kPrefetchWeightKBlocks */ 0,
        /* kSwapPipelineDecode */ false,
        /* kDistributedExpertBcast */ true,
        /* kQoQ */ $QQ,
        /* kDenseWeightTiles */ true,
        /* kHalfTileTasksRequested */ false,
        /* kSplitKL1Requested */ true,
        /* kL2HalfRowTasksRequested */ false,
        /* kSplitKL2Requested */ 0,
        /* kStreamKRequested */ ${STREAMK:-false},
        /* kNvlFastEpilogueRequested */ false,
        /* kFineCombineRequested */ true,
        /* kPushDispatchRequested */ true,
        /* kPushMaxTokensPerRank */ 2,
        /* kLeanRouting */ true,
        /* kPushDoneFlagsRequested */ true,
        /* kQoQInlineS2 */ $QIS2,
        /* kQoQInlineS2Frags */ 2,
        /* kQoQInlineS2Ilv */ false,
        /* kQoQInlineS2PrefetchPacked */ $QPF,
        /* kRFPrefetchPacked */ false,
        /* kStridedPoolDebug */ false,
        /* kL1TaskTiles */ $L1T,
        /* kL2TaskTiles */ $L2T,
        /* kHotSplitLevel */ ${HOT_SPLIT:-0}
    >);
};
CU
echo "=== $OUT ($(date -u +%FT%TZ))"
"$CUDA_HOME/bin/nvcc" -std=c++20 --diag-suppress=39,161,174,177,186,940 \
  --ptxas-options=--verbose,--warn-on-local-memory-usage \
  -I"$ROOT/deep_gemm/include" -I"$CUTLASS" --gpu-architecture=sm_90a \
  --compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi \
  -O3 --expt-relaxed-constexpr --expt-extended-lambda -cubin -o "$OUT/kernel.cubin" "$OUT/kernel.cu" \
  > "$OUT/nvcc.log" 2>&1
rc=$?
grep -E "error|Used [0-9]+ registers|spill|bytes stack|smem" "$OUT/nvcc.log" | grep -v "^$" | head -20
echo "COMPILE_RC=$rc"
exit $rc
