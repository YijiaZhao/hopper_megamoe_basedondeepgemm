#!/usr/bin/env bash
# K-part / MMA variant batch on 4 idle GPUs of an H20: for each config, ident (40 seeds, top-8
# equality vs the legacy 96 grid) then the standalone stamp/event timing (rows 1, mxfp4), one
# config per GPU in parallel. Configs: 97x4 wmma, 97x4 fma, 78x2 wmma, 78x2 fma.
# Usage (inside the container, repo root): [GPUS="0 1 2 3"] [OUT=...] bash scripts/run_fe78_variants.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
OUT=${OUT:-/raid/kimi/results/fe/fe78_variants}; mkdir -p "$OUT"
GPUS=${GPUS:-"0 1 2 3"}; read -r -a G <<< "$GPUS"
export PYTHONPATH="$ROOT" CUDA_HOME=/usr/local/cuda PYTHONUNBUFFERED=1
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
CONFIGS=("97 4 wmma" "97 4 fma" "auto 2 wmma" "auto 2 fma")
i=0
for cfg in "${CONFIGS[@]}"; do
  read -r GR KP MM <<< "$cfg"; gpu=${G[$((i % ${#G[@]}))]}; tag="g${GR}_k${KP}_${MM}"
  ( export CUDA_VISIBLE_DEVICES=$gpu DG_FE_TINYM_GRID=$GR DG_FE_TINYM_KPARTS=$KP DG_FE_TINYM_MMA=$MM
    timeout 900 python3 tests/test_frontend_fe78.py --seeds 40 --grid "$GR" --mma "$MM" > "$OUT/${tag}.ident.log" 2>&1
    timeout 900 python3 tests/fe_standalone_bench.py --quant mxfp4 --rows 1 > "$OUT/${tag}.r1_mxfp4.log" 2>&1
    timeout 900 python3 tests/fe_standalone_bench.py --quant mxfp4 --rows 2 > "$OUT/${tag}.r2_mxfp4.log" 2>&1 ) &
  i=$((i + 1))
done
wait
{ echo "=== FE78 variants $(git rev-parse --short HEAD) $(date)"; for f in "$OUT"/*.log; do echo "### $f"; cat "$f"; echo; done; } > "$OUT/VARIANTS.log"
echo "VARIANTS_DONE $(date)" >> "$OUT/VARIANTS.log"
