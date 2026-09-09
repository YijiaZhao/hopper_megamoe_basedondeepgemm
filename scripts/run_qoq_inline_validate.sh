#!/usr/bin/env bash
# Validate QoQ inline s2 (kQoQInlineS2, DG_FP4_QOQ_INLINE_S2): correctness for QoQ at
# T = 2 8 8 16 tokens per rank with the knob ON, QoQ T=8 with the knob OFF and MXFP4
# T=8 as controls; then the phase-stamp probe: qoq knob 1 twice, knob 0 once, mxfp4
# once, M = 2 8 16 global tokens. Waits for idle GPUs (no compute apps, no nvcc)
# before every GPU run.
# Usage (inside four_api_build container): [TOKENS="2 8 8 16"] bash scripts/run_qoq_inline_validate.sh [corr|perf|all] [tag]
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
LOG=corr_qis2$TAG.log

wait_idle() {
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
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
  for T in ${TOKENS:-2 8 8 16}; do
    run_corr qoq_mega_moe_fused $T DG_FP4_QOQ_INLINE_S2=1
  done
  run_corr qoq_mega_moe_fused 8 DG_FP4_QOQ_INLINE_S2=0
  run_corr mxfp4_mega_moe_fused 8 DG_FP4_QOQ_INLINE_S2=1
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  wait_idle; DG_FP4_QOQ_INLINE_S2=1 LOG_TAG=_qis2ON${TAG}1 bash scripts/run_probe.sh qoq 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_QOQ_INLINE_S2=0 LOG_TAG=_qis2OFF${TAG}1 bash scripts/run_probe.sh qoq 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_QOQ_INLINE_S2=1 LOG_TAG=_qis2ON${TAG}2 bash scripts/run_probe.sh qoq 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_QOQ_INLINE_S2=1 LOG_TAG=_qis2${TAG} bash scripts/run_probe.sh mxfp4 8 > /dev/null 2>&1
  echo ALL_PERF_DONE >> "$LOG"
fi
