#!/bin/bash
# Rebuild the extension in this worktree (run inside four_api_build from the repo root, usual env).
# Usage: bash tests/build_ccrouter.sh [logfile]
cd "$(dirname "$0")/.."
LOG=${1:-/raid/kimi/results/fe_cc4/build.log}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-/raid/kimi/dg_dev/third-party/cutlass/include}
if [ ! -e third-party/cutlass/include/cute ]; then rmdir third-party/cutlass 2>/dev/null || rm -rf third-party/cutlass; ln -sfn "$(dirname "$DG_CUTLASS_INCLUDE_PATH")" third-party/cutlass; fi
ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cutlass" deep_gemm/include/cutlass
ln -sfn "$DG_CUTLASS_INCLUDE_PATH/cute" deep_gemm/include/cute
echo "BUILD_START $(date +%T) $(git log --oneline -1)" > "$LOG"
rm -rf build
python3 setup.py build >> "$LOG" 2>&1
so=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so" ]; then ln -sf "../$so" deep_gemm/; echo "BUILD_OK $(date +%T) $so" >> "$LOG"; else echo "BUILD_FAILED $(date +%T)" >> "$LOG"; fi
grep -E "error|Error" "$LOG" | grep -v "Werror\|-Wno-error\|no-error" | head -20 >> "$LOG.err"
