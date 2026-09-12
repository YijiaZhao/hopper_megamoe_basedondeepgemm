#!/usr/bin/env bash
# NCU of the cc router (router_cc_lean_kernel) in the pipeline configuration (DG_FE_SELECT_IN_MEGA=1: router +
# keys only) and the standalone configuration (ticket + last-arriver select). --set full --clock-control none,
# default cache control (L2 flushed between passes, i.e. cold code + cold weights, as after the Mega weight
# stream). One GPU.
# Usage (in the build container, repo root): NCU_FE_GPU=7 bash tests/ncu_fe5.sh [outdir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/raid/kimi/results/fe5/ncu}; GPU=${NCU_FE_GPU:-7}; WARM=${NCU_FE_WARMUP:-5}
mkdir -p "$OUT"; export CUDA_VISIBLE_DEVICES=$GPU
nvidia-smi -i "$GPU" --query-gpu=index,name,clocks.sm,clocks.mem,memory.total --format=csv > "$OUT/gpu.txt"
run_one() {  # selmega quant rows tag
  echo "=== $4 ($(date +%T))"
  DG_FE_SELECT_IN_MEGA=$1 timeout 900 ncu --target-processes application-only \
    --kernel-name "regex:router_(quant_topk|cc_lean)_kernel" --launch-skip "$WARM" --launch-count 1 \
    --set full --import-source no --clock-control none -f -o "$OUT/$4" \
    python3 tests/ncu_frontend_tinym.py --quant "$2" --rows "$3" --warmup "$WARM" --l2-flush 1 > "$OUT/$4.run.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$4.run.log"
  if [ -f "$OUT/$4.ncu-rep" ]; then
    ncu --import "$OUT/$4.ncu-rep" --page details > "$OUT/$4.details.txt" 2>&1
    ncu --import "$OUT/$4.ncu-rep" --page raw --csv > "$OUT/$4.raw.csv" 2>&1
  fi
}
run_one 1 mxfp4 1 sm1_rows1_mxfp4
run_one 1 qoq 2 sm1_rows2_qoq
run_one 0 mxfp4 1 sm0_rows1_mxfp4
run_one 0 qoq 2 sm0_rows2_qoq
echo "NCU_ALL_DONE $(date +%T)"
