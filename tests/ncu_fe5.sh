#!/usr/bin/env bash
# Round-5 NCU of the cc router, baseline vs DG_FE_CC_LEAN=1, in the pipeline configuration
# (DG_FE_SELECT_IN_MEGA=1: router + keys only) and the knob-0 configuration (ticket + last-arriver select,
# insert vs pruned). --set full --clock-control none, default cache control (L2 flushed between passes,
# i.e. cold code + cold weights, as after the Mega weight stream). One GPU.
# Usage (in four_api_build, repo root): NCU_FE_GPU=7 bash tests/ncu_fe5.sh [outdir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/raid/kimi/results/fe5/ncu}; GPU=${NCU_FE_GPU:-7}; WARM=${NCU_FE_WARMUP:-5}
mkdir -p "$OUT"; export CUDA_VISIBLE_DEVICES=$GPU
export DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc
nvidia-smi -i "$GPU" --query-gpu=index,name,clocks.sm,clocks.mem,memory.total --format=csv > "$OUT/gpu.txt"
run_one() {  # selmega lean sel quant rows tag
  echo "=== $6 ($(date +%T))"
  DG_FE_SELECT_IN_MEGA=$1 DG_FE_CC_LEAN=$2 DG_FE_CC_SELECT=$3 timeout 900 ncu --target-processes application-only \
    --kernel-name "regex:router_(quant_topk|cc_lean)_kernel" --launch-skip "$WARM" --launch-count 1 \
    --set full --import-source no --clock-control none -f -o "$OUT/$6" \
    python3 tests/ncu_frontend_tinym.py --quant "$4" --rows "$5" --tinym 1 --warmup "$WARM" --grid auto --mma cc --l2-flush 1 > "$OUT/$6.run.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$6.run.log"
  if [ -f "$OUT/$6.ncu-rep" ]; then
    ncu --import "$OUT/$6.ncu-rep" --page details > "$OUT/$6.details.txt" 2>&1
    ncu --import "$OUT/$6.ncu-rep" --page raw --csv > "$OUT/$6.raw.csv" 2>&1
  fi
}
run_one 1 0 insert mxfp4 1 sm1_base_rows1_mxfp4
run_one 1 1 insert mxfp4 1 sm1_lean_rows1_mxfp4
run_one 1 0 insert qoq 2 sm1_base_rows2_qoq
run_one 1 1 insert qoq 2 sm1_lean_rows2_qoq
run_one 0 0 insert mxfp4 1 sm0_base_insert_rows1_mxfp4
run_one 0 0 pruned mxfp4 1 sm0_base_pruned_rows1_mxfp4
run_one 0 1 pruned mxfp4 1 sm0_lean_pruned_rows1_mxfp4
echo "NCU_ALL_DONE $(date +%T)"
