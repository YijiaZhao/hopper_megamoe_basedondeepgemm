#!/bin/bash
# Build this worktree's extension and run the DG_FE_TINYM_MMA=tc16 matrix on ONE GPU:
# standalone FE stamps + CUDA-event time for rows 1|2 x mxfp4|qoq, 96x4 WMMA vs cc vs tc16 (6 warps/unit)
# vs tc16w3 (3 warps/unit), each with and without DG_FE_ROUTER_L2_PERSIST=1, then the equality test
# vs the 96x4 WMMA reference (tests/test_frontend_fe78.py, SEEDS seeds x rows 1,2 x mxfp4,qoq).
# Usage (host): docker exec -e CUDA_VISIBLE_DEVICES=k ... -w <worktree> four_api_build bash tests/run_tc16_matrix.sh [build=1] [seeds=1000] [mmas="cc tc16 tc16w3"]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-/raid/kimi/dg_dev/third-party/cutlass/include}
if [ ! -e third-party/cutlass/include/cute ]; then rmdir third-party/cutlass 2>/dev/null || rm -rf third-party/cutlass; ln -sfn "$(dirname "$DG_CUTLASS_INCLUDE_PATH")" third-party/cutlass; fi
[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "pin CUDA_VISIBLE_DEVICES"; exit 1; }
echo "using GPU $CUDA_VISIBLE_DEVICES"
if [ "${1:-1}" = "1" ]; then
  ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cutlass" deep_gemm/include/cutlass
  ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cute" deep_gemm/include/cute
  rm -rf build; python3 setup.py build 2>&1 | grep -E "error|warning: #|Error|registers|spill" | head -60 || true
  so=$(find build -name "*.so" -type f | head -n 1); [ -n "$so" ] || { echo BUILD FAILED; exit 1; }
  ln -sf "../$so" deep_gemm/; echo "BUILD OK $so"
fi
SEEDS="${2:-1000}"; MMAS="${3:-cc tc16 tc16w3}"
for rows in 1 2; do for q in mxfp4 qoq; do
  echo "### wmma96 rows=$rows quant=$q"; timeout 600 python3 tests/fe_standalone_bench.py --quant $q --rows $rows --grid 96 --mma wmma 2>&1 | grep -v "^  CTA placement"
  for persist in 0 1; do for mma in $MMAS; do
    echo "### $mma persist=$persist rows=$rows quant=$q"
    DG_FE_ROUTER_L2_PERSIST=$persist DG_FE_TINYM_GRID=auto timeout 600 python3 tests/fe_standalone_bench.py --quant $q --rows $rows --grid auto --mma $mma 2>&1 | grep -v "^  CTA placement"
  done; done
done; done
for mma in $MMAS; do
  echo "### equality wmma(96) vs $mma"; timeout 900 python3 tests/test_frontend_fe78.py --seeds $SEEDS --rows 1 2 --grid auto --mma $mma 2>&1 | tail -4
done
echo MATRIX DONE
