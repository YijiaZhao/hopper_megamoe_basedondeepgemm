#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/raid/kimi/DeepGEMM_four_api_h20}
OUT=${OUT:-/raid/kimi/megamoe_opt2_results/four_api_delivery_8cf5472_20260903}
mkdir -p "$OUT"
cd "$ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DG_BENCH_FLUSH_L2_BYTES=268435456
export PYTHONUNBUFFERED=1
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
COMMON=(--trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true)
run() {
  local name=$1; shift
  echo "=== $name ==="
  nsys profile "${COMMON[@]}" --output="$OUT/$name" "$@"
}
for scope in e2e mega; do
  for backend in split fused; do
    for quant in mxfp4 qoq; do
      for m in 2 8 16; do
        run "${scope}_${backend}_${quant}_M${m}" \
          torchrun --standalone --nproc_per_node=8 tests/profile_four_api_h20.py \
          --scope "$scope" --backend "$backend" --quant "$quant" --global-tokens "$m"
      done
    done
  done
done
find "$OUT" -maxdepth 1 -type f -name '*.nsys-rep' -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
count=$(find "$OUT" -maxdepth 1 -type f -name '*.nsys-rep' | wc -l)
echo "TIMELINE_COUNT=$count"
test "$count" -eq 24
