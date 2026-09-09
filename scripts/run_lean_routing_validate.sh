#!/usr/bin/env bash
# Validate lean routing (kLeanRouting, DG_FP4_LEAN_ROUTING): correctness for MXFP4 + QoQ
# at T = 2 8 8 16 tokens per rank (push dispatch path) and MXFP4 at T = 128 512 (pull
# path; routing is shared with large M), run twice back-to-back; then the phase-stamp
# probe LEAN=1 / LEAN=0 / LEAN=1 (same session), mxfp4, M = 2 8 16 global tokens.
# Waits for idle GPUs (no compute apps, no live nvcc) before every GPU run.
# Usage (inside four_api_build container): bash scripts/run_lean_routing_validate.sh [corr|perf|all] [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}
TAG=${2:-}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$ROOT/.jit_cache}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
TR=/usr/local/bin/torchrun
LOG=corr_lean$TAG.log

wait_idle() {
  local waited=0
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    m=$(ps -C nvcc -o stat= 2>/dev/null | grep -v '^Z' | grep -c .)
    [ "$n" = 0 ] && [ "$m" = 0 ] && break
    sleep 5; waited=$((waited + 5))
    [ $waited -ge 600 ] && { echo "wait_idle: still busy after 600s, retrying" >> "$LOG"; waited=0; }
  done
}

run_corr() {  # $1 = apis (quoted list), $2 = T, $3.. = env assignments
  local apis=$1 T=$2; shift 2
  wait_idle
  echo "--- apis=[$apis] T=$T ($*)" >> "$LOG"
  env "$@" timeout 900 $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis $apis --tokens "$T" >> "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  : > "$LOG"
  for pass in 1 2; do
    echo "=== correctness pass $pass" >> "$LOG"
    for T in ${TOKENS:-2 8 8 16}; do
      run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" $T DG_FP4_LEAN_ROUTING=1
    done
    for T in ${TOKENS_LARGE:-128 512}; do
      run_corr "mxfp4_mega_moe_fused" $T DG_FP4_LEAN_ROUTING=1
    done
  done
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  for q in ${QUANTS:-mxfp4}; do
    wait_idle; DG_FP4_LEAN_ROUTING=1 LOG_TAG=_leanON${TAG}1 bash scripts/run_probe.sh $q 2 8 16 > /dev/null 2>&1
    wait_idle; DG_FP4_LEAN_ROUTING=0 LOG_TAG=_leanOFF${TAG}1 bash scripts/run_probe.sh $q 2 8 16 > /dev/null 2>&1
    wait_idle; DG_FP4_LEAN_ROUTING=1 LOG_TAG=_leanON${TAG}2 bash scripts/run_probe.sh $q 2 8 16 > /dev/null 2>&1
  done
  echo ALL_PERF_DONE >> "$LOG"
fi
