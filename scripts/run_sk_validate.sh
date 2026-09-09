#!/usr/bin/env bash
# Validate the stream-K knob (DG_FP4_STREAMK): correctness (race-sensitive tiny T
# repeated, then 128/512 which must stay on the wave scheduler, then QoQ), then the
# phase-stamp probe (ON / OFF / ON in one session, then QoQ ON). Waits for idle GPUs
# first (the .7 box is shared).
# Usage (inside four_api_build container): bash scripts/run_sk_validate.sh [corr|perf|all]
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
LOG=corr_sk.log

wait_idle() {  # wait (<= 60 min) until no compute process / nvcc is running
  for _ in $(seq 1 360); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && ! pgrep -x nvcc > /dev/null; then
      return 0
    fi
    sleep 10
  done
  return 1
}

run_corr() {  # $1.. = env assignments
  echo "=== STREAMK ($*)" >> "$LOG"
  for T in 2 8 8 16 16; do
    echo "--- mxfp4+qoq T=$T" >> "$LOG"
    wait_idle
    env "$@" $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1
    echo "EXIT=$?" >> "$LOG"
  done
  for T in 128 512; do
    echo "--- mxfp4 T=$T" >> "$LOG"
    wait_idle
    env "$@" $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis mxfp4_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1
    echo "EXIT=$?" >> "$LOG"
  done
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  : > "$LOG"
  run_corr DG_FP4_STREAMK=1
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  wait_idle; DG_FP4_STREAMK=1 LOG_TAG=_sk1a bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_STREAMK=0 LOG_TAG=_sk0  bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_STREAMK=1 LOG_TAG=_sk1b bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_STREAMK=1 LOG_TAG=_sk1  bash scripts/run_probe.sh qoq 2 8 16 > /dev/null 2>&1
  echo ALL_PERF_DONE >> "$LOG"
fi
