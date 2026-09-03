#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/raid/kimi/DeepGEMM_four_api_h20}
MODE=${1:-all}
cd "$ROOT"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1

# The four explicit APIs must select their own format.  Keep legacy mode
# selectors out of the validation environment.
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW

build() {
  bash develop.sh
}

smoke() {
  torchrun --standalone --nproc_per_node=8 tests/test_four_api_smoke.py
}

correctness() {
  torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py
}

case "$MODE" in
  build) build ;;
  smoke) smoke ;;
  correctness) correctness ;;
  all) build; smoke; correctness ;;
  *) echo "usage: $0 [build|smoke|correctness|all]" >&2; exit 2 ;;
esac
