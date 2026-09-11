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
if [ "${1:-1}" = "1" ]; then
  ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cutlass" deep_gemm/include/cutlass
  ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cute" deep_gemm/include/cute
  rm -rf build; python3 setup.py build 2>&1 | grep -E "error|warning: #|Error" | head -40 || true
  so=$(find build -name "*.so" -type f | head -n 1); [ -n "$so" ] || { echo BUILD FAILED; exit 1; }
  ln -sf "../$so" deep_gemm/; echo "BUILD OK $so"
fi
for rows in 1 2; do for q in mxfp4 qoq; do
  echo "### default rows=$rows quant=$q"; timeout 600 python3 tests/fe_standalone_bench.py --quant $q --rows $rows --grid 96 --mma wmma 2>&1 | grep -v "^  CTA placement"
  for mma in cc cc6; do
    echo "### $mma rows=$rows quant=$q"; DG_FE_TINYM_GRID=auto timeout 600 python3 tests/fe_standalone_bench.py --quant $q --rows $rows --grid auto --mma $mma 2>&1 | grep -v "^  CTA placement"
  done
done; done
echo "### equality wmma(96) vs cc"; timeout 600 python3 tests/test_frontend_fe78.py --seeds 500 --rows 1 2 --grid auto --mma cc 2>&1 | tail -4
echo "### equality wmma(96) vs cc6"; timeout 600 python3 tests/test_frontend_fe78.py --seeds 100 --rows 1 2 --grid auto --mma cc6 2>&1 | tail -3
