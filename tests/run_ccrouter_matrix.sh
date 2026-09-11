#!/bin/bash
# Build the extension in this worktree and run the DG_FE_TINYM_MMA=cc matrix (single GPU):
# standalone FE stamps/event for rows 1|2 x mxfp4|qoq, default (96 x 4 WMMA) vs cc vs cc6,
# then the WMMA-vs-cc equality test (tests/test_frontend_fe78.py, 500 seeds x rows 1,2 = 1500 rows).
# Usage (host): docker exec -e CUDA_VISIBLE_DEVICES=k ... four_api_build bash tests/run_ccrouter_matrix.sh [build=1]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-/raid/kimi/dg_dev/third-party/cutlass/include}
# setup.py hard-codes third-party/cutlass: point the (empty submodule) directory at the shared checkout
if [ ! -e third-party/cutlass/include/cute ]; then rmdir third-party/cutlass 2>/dev/null || rm -rf third-party/cutlass; ln -sfn "$(dirname "$DG_CUTLASS_INCLUDE_PATH")" third-party/cutlass; fi
# pick one idle GPU (other agents run on this box) unless the caller pinned one
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 == 0 && $3 < 200 {g=$1} END {print g}')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "no idle GPU"; exit 1; }
fi
echo "using GPU $CUDA_VISIBLE_DEVICES"
if [ "${1:-1}" = "1" ]; then
  ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cutlass" deep_gemm/include/cutlass
  ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cute" deep_gemm/include/cute
  rm -rf build; python3 setup.py build 2>&1 | grep -E "error|warning: #|Error" | head -40 || true
  so=$(find build -name "*.so" -type f | head -n 1); [ -n "$so" ] || { echo BUILD FAILED; exit 1; }
  ln -sf "../$so" deep_gemm/; echo "BUILD OK $so"
fi
MMAS="${2:-cc cc6}"; TAG="${3:-}"; SEEDS="${4:-500}"; GRID="${GRID:-auto}"
for rows in 1 2; do for q in mxfp4 qoq; do
  [ -z "$TAG" ] && { echo "### default rows=$rows quant=$q"; timeout 600 python3 tests/fe_standalone_bench.py --quant $q --rows $rows --grid 96 --mma wmma 2>&1 | grep -v "^  CTA placement"; }
  for mma in $MMAS; do
    echo "### $mma$TAG rows=$rows quant=$q"; DG_FE_TINYM_GRID=auto timeout 600 python3 tests/fe_standalone_bench.py --quant $q --rows $rows --grid $GRID --mma $mma 2>&1 | grep -v "^  CTA placement"
  done
done; done
for mma in $MMAS; do
  echo "### equality wmma(96) vs $mma$TAG"; timeout 600 python3 tests/test_frontend_fe78.py --seeds $SEEDS --rows 1 2 --grid $GRID --mma $mma 2>&1 | tail -4
done
