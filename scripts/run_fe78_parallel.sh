#!/usr/bin/env bash
# Parallel single-GPU FE matrix on an idle 8-GPU H20: {wmma,fma} x {rows 1,2} x {mxfp4,qoq} = 8
# configs, one per GPU listed in GPUS (default 0..6 -> two rounds; GPU 7 is left for NCU).
# Writes $OUT/<mma>_r<rows>_<quant>.log and a combined $OUT/MATRIX.log.
# Usage (inside the build container, repo root): [GPUS="0 1 2 3 4 5 6"] [OUT=...] bash scripts/run_fe78_parallel.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
OUT=${OUT:-/raid/kimi/results/fe/fe78_matrix}; mkdir -p "$OUT"
GPUS=${GPUS:-"0 1 2 3 4 5 6"}; read -r -a G <<< "$GPUS"
ITERS=${ITERS:-100}
export PYTHONPATH="$ROOT" CUDA_HOME=/usr/local/cuda PYTHONUNBUFFERED=1
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
CONFIGS=(); for MM in wmma fma; do for R in 1 2; do for Q in mxfp4 qoq; do CONFIGS+=("$MM $R $Q"); done; done; done
i=0; pids=()
for cfg in "${CONFIGS[@]}"; do
  read -r MM R Q <<< "$cfg"; gpu=${G[$((i % ${#G[@]}))]}
  if [ $i -ge ${#G[@]} ]; then wait "${pids[$((i - ${#G[@]}))]}"; fi     # second round waits for its GPU
  CUDA_VISIBLE_DEVICES=$gpu DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=$MM timeout 900 \
    python3 tests/fe_standalone_bench.py --quant "$Q" --rows "$R" --iters "$ITERS" > "$OUT/${MM}_r${R}_${Q}.log" 2>&1 &
  pids+=($!); i=$((i + 1))
done
wait
{ echo "=== FE78 parallel matrix $(git rev-parse --short HEAD) $(date)"; for cfg in "${CONFIGS[@]}"; do read -r MM R Q <<< "$cfg"; cat "$OUT/${MM}_r${R}_${Q}.log"; echo; done; } > "$OUT/MATRIX.log"
echo "MATRIX_DONE $(date)" >> "$OUT/MATRIX.log"
