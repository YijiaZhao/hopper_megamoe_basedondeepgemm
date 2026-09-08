#!/usr/bin/env bash
# Validate the L2 split-K (ITEM A) and fast NVLink-barrier epilogue (ITEM B) knobs
# separately: correctness (race-sensitive small T repeated, then 128/512, then QoQ)
# per item, then the phase-stamp probe twice per item (and twice with both ON).
# Usage (inside four_api_build container): bash scripts/run_l2sk_validate.sh [corr|perf|all]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
TR=/usr/local/bin/torchrun
LOG=corr_l2sk.log

run_corr() {  # $1 = item tag, $2.. = env assignments
  local tag=$1; shift
  echo "=== ITEM $tag ($*)" >> "$LOG"
  for T in 2 8 8 8 16 16 128 512; do
    echo "--- mxfp4 T=$T" >> "$LOG"
    env "$@" $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis mxfp4_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1
    echo "EXIT=$?" >> "$LOG"
  done
  echo "--- qoq T=8" >> "$LOG"
  env "$@" $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis qoq_mega_moe_fused --tokens 8 >> "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
}

if [ "$MODE" = corr ] || [ "$MODE" = all ] || [ "$MODE" = corrA ]; then
  : > "$LOG"
  run_corr A DG_FP4_NVL_FAST_EPI=0 DG_FP4_SPLITK_L2=1
  [ "$MODE" = corrA ] || run_corr B DG_FP4_SPLITK_L2=0 DG_FP4_NVL_FAST_EPI=1
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  # Round-robin over configs to cancel session drift: OFF / A / B / AB, 3 reps.
  for rep in 1 2 3; do
    DG_FP4_SPLITK_L2=0 DG_FP4_NVL_FAST_EPI=0 LOG_TAG=_l2skOFF$rep bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
    DG_FP4_SPLITK_L2=1 DG_FP4_NVL_FAST_EPI=0 LOG_TAG=_l2skA$rep bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
    DG_FP4_SPLITK_L2=0 DG_FP4_NVL_FAST_EPI=1 LOG_TAG=_l2skB$rep bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
    DG_FP4_SPLITK_L2=1 DG_FP4_NVL_FAST_EPI=1 LOG_TAG=_l2skAB$rep bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  done
  echo ALL_PERF_DONE >> "$LOG"
fi
