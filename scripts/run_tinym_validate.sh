#!/usr/bin/env bash
# Validate the tiny-M CUDA-core GEMV path (kTinyMGemv, DG_FP4_TINYM): correctness for
# MXFP4 + QoQ at T = 2 8 8 16 tokens per rank (the path forced on via a large
# DG_FP4_TINYM_MAX_M so the multi-pool-block cases are exercised), then the phase-stamp
# probe TINYM=1 vs 0, mxfp4 + qoq, M = 2 8 16 global tokens, two interleaved reps.
# Waits for idle GPUs (no compute apps, no nvcc) before every GPU run.
# Usage (inside four_api_build container): [APIS="..."] bash scripts/run_tinym_validate.sh [corr|perf|all] [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}
TAG=${2:-}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
TR=/usr/local/bin/torchrun
LOG=corr_tinym$TAG.log

wait_idle() {
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    # live nvcc only (defunct/zombie nvcc entries linger when a parent never reaps them)
    m=$(ps -C nvcc -o stat= 2>/dev/null | grep -v '^Z' | grep -c .)
    [ "$n" = 0 ] && [ "$m" = 0 ] && break
    sleep 5
  done
}

run_corr() {  # $1 = api, $2 = T, $3.. = env assignments
  local api=$1 T=$2; shift 2
  wait_idle
  echo "--- $api T=$T ($*)" >> "$LOG"
  env "$@" timeout 900 $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis "$api" --tokens "$T" >> "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  : > "$LOG"
  for api in ${APIS:-mxfp4_mega_moe_fused qoq_mega_moe_fused}; do
    for T in 2 8 8 16; do
      run_corr $api $T DG_FP4_TINYM=1 DG_FP4_TINYM_MAX_M=4096
    done
  done
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  for rep in 1 2; do
    for q in mxfp4 qoq; do
      wait_idle; DG_FP4_TINYM=1 LOG_TAG=_tinymON${TAG}$rep bash scripts/run_probe.sh $q 2 8 16 > /dev/null 2>&1
      wait_idle; DG_FP4_TINYM=0 LOG_TAG=_tinymOFF${TAG}$rep bash scripts/run_probe.sh $q 2 8 16 > /dev/null 2>&1
    done
  done
  echo ALL_PERF_DONE >> "$LOG"
fi
