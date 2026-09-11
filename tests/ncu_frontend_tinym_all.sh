#!/usr/bin/env bash
# Nsight Compute matrix for the Fable frontend kernel, standalone on ONE GPU (no torchrun):
#   tinym (DG_FE_TINYM=1) x {mxfp4, qoq} x global M {2, 4, 8, 16} + legacy (DG_FE_TINYM=0) x {mxfp4, qoq} x M 8.
# Reports are named by global M; the kernel only sees rank 0's rows per the E2E layout:
# M2 -> 1 row, M4 -> 1, M8 -> 1, M16 -> 2 (all four M run even where inputs coincide).
# Exports per report: .details.txt, .details.csv, .raw.csv; then SUMMARY.md.
# Run inside four_api_build from the repo root with the usual env (CUDA_HOME, PYTHONPATH, ...).
# Env: NCU_FE_OUT (default /raid/kimi/results/ncu_fe), NCU_FE_GPU (default 7), NCU_FE_WARMUP (5).
set -uo pipefail
OUT=${NCU_FE_OUT:-/raid/kimi/results/ncu_fe}
GPU=${NCU_FE_GPU:-7}
WARM=${NCU_FE_WARMUP:-5}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
nvidia-smi -i "$GPU" --query-gpu=index,name,clocks.sm,clocks.mem --format=csv > "$OUT/gpu_clocks_before.txt"
rows_for_m() { if [ "$1" -ge 16 ]; then echo 2; else echo 1; fi; }
run_one() {  # tinym quant M [grid]
  local rows; rows=$(rows_for_m "$3")
  local grid=${4:-96}
  local tag="fe_$([ "$1" = 1 ] && echo tinym || echo legacy)_M$3_$2"
  [ "$1" = 1 ] && [ "$grid" != 96 ] && tag="fe_fullk${grid}_M$3_$2"
  echo "=== $tag ($(date +%T))"
  DG_FE_TINYM=$1 timeout 900 ncu --target-processes application-only \
    --kernel-name regex:router_quant_topk_kernel --launch-skip "$WARM" --launch-count 1 \
    --set full --import-source no --clock-control none -f -o "$OUT/$tag" \
    python3 tests/ncu_frontend_tinym.py --quant "$2" --rows "$rows" --tinym "$1" --warmup "$WARM" --grid "$grid" \
    > "$OUT/$tag.run.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$tag.run.log"
  if [ -f "$OUT/$tag.ncu-rep" ]; then
    ncu --import "$OUT/$tag.ncu-rep" --page details > "$OUT/$tag.details.txt" 2>&1
    ncu --import "$OUT/$tag.ncu-rep" --page details --csv > "$OUT/$tag.details.csv" 2>&1
    ncu --import "$OUT/$tag.ncu-rep" --page raw --csv > "$OUT/$tag.raw.csv" 2>&1
  fi
}
GRIDS=${NCU_FE_GRIDS:-"96 auto"}
for g in $GRIDS; do for q in mxfp4 qoq; do for m in 2 4 8 16; do run_one 1 "$q" "$m" "$g"; done; done; done
for q in mxfp4 qoq; do run_one 0 "$q" 8; done
nvidia-smi -i "$GPU" --query-gpu=index,clocks.sm,clocks.mem --format=csv > "$OUT/gpu_clocks_after.txt"
python3 tests/ncu_fe_summary.py "$OUT" > "$OUT/SUMMARY.md"
echo "ALL_DONE $(date +%T)"
