#!/usr/bin/env bash
# Reconcile the in-kernel phase-stamp probe against the official nsys capture for one
# (quant, M) of Mega-only Fused, back to back on the same build (run inside four_api_build):
#   1. official : nsys profile ... tests/profile_four_api_h20.py --scope mega --backend fused
#   2. probe    : tests/profile_fused_phase_stamps.py (PROBE_DUMP=1 -> one line per iteration)
#   3. probe under the same nsys flags (same launches seen by globaltimer AND by CUPTI)
#   4. probe --no-stamps (CUDA-event wall only, stamp-overhead check)
# Usage: bash scripts/reconcile_probe_vs_official.sh OUTDIR quant M [tag]
# Then:  python3 scripts/reconcile_nsys_devices.py OUTDIR/*.nsys-rep
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
OUT=$1; QUANT=$2; M=$3; TAG=${4:-a}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
mkdir -p "$OUT"
name="${QUANT}_M${M}_${TAG}"

wait_idle() {
  local i
  for i in $(seq 1 150); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then return 0; fi
    sleep 2
  done
  echo "GPUs still busy after 300 s" >&2; return 1
}
NSYS=(/usr/local/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none
      --cpuctxsw=none --force-overwrite=true)
TR=(/usr/local/bin/torchrun --standalone --nproc_per_node=8)
PROBE=(tests/profile_fused_phase_stamps.py --quant "$QUANT" --global-tokens "$M" --iters 20)

echo "=== build: $(git rev-parse --short HEAD) clocks: $(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits | tr '\n' ' ')"
wait_idle
timeout 600 "${NSYS[@]}" --output="$OUT/official_$name" "${TR[@]}" tests/profile_four_api_h20.py \
  --scope mega --backend fused --quant "$QUANT" --global-tokens "$M" > "$OUT/official_$name.log" 2>&1
echo "official EXIT=$?"
wait_idle
PROBE_DUMP=1 timeout 600 "${TR[@]}" "${PROBE[@]}" > "$OUT/probe_$name.log" 2>&1
echo "probe EXIT=$?"
wait_idle
PROBE_DUMP=1 timeout 600 "${NSYS[@]}" --output="$OUT/probe_nsys_$name" "${TR[@]}" "${PROBE[@]}" \
  > "$OUT/probe_nsys_$name.log" 2>&1
echo "probe+nsys EXIT=$?"
wait_idle
timeout 600 "${TR[@]}" "${PROBE[@]}" --no-stamps > "$OUT/probe_nostamps_$name.log" 2>&1
echo "probe --no-stamps EXIT=$?"
echo "RECONCILE_DONE $name"
