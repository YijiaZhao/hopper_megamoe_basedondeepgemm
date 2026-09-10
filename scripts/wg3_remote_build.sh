#!/usr/bin/env bash
# Build the deep_gemm C extension in this worktree (third-party submodules borrowed from
# /raid/kimi/dg_dev), then run scripts/wg3_validate.sh MODE (default corr).
#   nohup bash scripts/wg3_remote_build.sh > /raid/kimi/wg3_build.log 2>&1 &
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
DEV=${DEV:-/raid/kimi/dg_dev}
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0a}
for d in cutlass fmt; do
  if [ ! -e "third-party/$d/include" ]; then
    rmdir "third-party/$d" 2>/dev/null
    ln -sfn "$DEV/third-party/$d" "third-party/$d"
  fi
done
ln -sfn "$DEV/third-party/cutlass/include/cutlass" deep_gemm/include/cutlass
ln -sfn "$DEV/third-party/cutlass/include/cute" deep_gemm/include/cute
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  rm -rf build
  echo "BUILD start $(date -u +%FT%TZ) $(git rev-parse --short HEAD)"
  taskset -c "${CORES:-64-95}" python setup.py build || { echo BUILD_FAILED; exit 1; }
  so=$(find build -name '*.so' -type f | head -n 1)
  [ -n "$so" ] || { echo "BUILD_FAILED (no .so)"; exit 1; }
  ln -sf "../$so" deep_gemm/
  echo "BUILD_OK $(date -u +%FT%TZ) $so"
fi
MODE=${MODE:-corr} bash scripts/wg3_validate.sh
echo "REMOTE_BUILD_DONE $MODE"
