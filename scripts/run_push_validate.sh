#!/usr/bin/env bash
# Validate push dispatch (kPushDispatch, DG_FP4_PUSH_DISPATCH): build, correctness
# (mxfp4 + qoq T = 2 8 8 16 twice back-to-back, mxfp4 T = 128 512 == pull path), a
# 200-iteration graph-replay stress probe at M = 8, then the phase-stamp probe
# PUSH=1 (x2) vs PUSH=0 (x1) for mxfp4 M = 2 8 16 and qoq M = 8.
# Waits for idle GPUs (no compute apps, no live nvcc) before every GPU run.
# Usage (inside four_api_build container): bash scripts/run_push_validate.sh [build|corr|stress|perf|all] [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}
TAG=${2:-}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export TORCH_CUDA_ARCH_LIST=9.0a
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_JIT_CACHE_DIR="$ROOT/.jit_cache"
TR=/usr/local/bin/torchrun
LOG=push_validate$TAG.log

wait_idle() {
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    m=$(ps -C nvcc -o stat= 2>/dev/null | grep -v '^Z' | grep -c .)
    [ "$n" = 0 ] && [ "$m" = 0 ] && break
    sleep 5
  done
}

run_corr() {  # $1 = api list (space separated), $2 = token list (one launch per T), $3.. = env assignments
  local apis=$1 toks=$2 T; shift 2
  for T in $toks; do
    wait_idle
    echo "--- apis=[$apis] tokens=$T ($*)" >> "$LOG"
    env "$@" timeout ${CORR_TIMEOUT:-1200} $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis $apis --tokens $T >> "$LOG" 2>&1
    echo "EXIT=$?" >> "$LOG"
  done
}

if [ "$MODE" = isolate ]; then
  # Hang isolation with spin-wait timeouts (DG_FP4_SPIN_TIMEOUT=1 -> trap + site tag):
  # push T=2 (mxfp4 only), then the pull path with the same layout.
  LOG=push_isolate$TAG.log; : > "$LOG"
  run_corr "mxfp4_mega_moe_fused" "2" DG_FP4_PUSH_DISPATCH=1 DG_FP4_SPIN_TIMEOUT=1
  run_corr "mxfp4_mega_moe_fused" "2" DG_FP4_PUSH_DISPATCH=0 DG_FP4_SPIN_TIMEOUT=1
  echo ALL_ISOLATE_DONE >> "$LOG"
  exit 0
fi

if [ "$MODE" = build ] || [ "$MODE" = all ]; then
  : > "$LOG"
  echo "--- build" >> "$LOG"
  bash develop.sh >> "$LOG" 2>&1
  echo "BUILD_EXIT=$?" >> "$LOG"
fi

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  [ "$MODE" = corr ] && : > "$LOG"
  run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" "2 8 8 16" DG_FP4_PUSH_DISPATCH=1
  run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" "2 8 8 16" DG_FP4_PUSH_DISPATCH=1
  run_corr "mxfp4_mega_moe_fused" "128 512" DG_FP4_PUSH_DISPATCH=1
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = stress ] || [ "$MODE" = all ]; then
  wait_idle
  echo "--- stress mxfp4 M=8 iters=200 PUSH=1" >> "$LOG"
  DG_FP4_PUSH_DISPATCH=1 timeout 1200 $TR --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py \
    --quant mxfp4 --global-tokens 8 --iters 200 > "probe_mxfp4_m8_stress$TAG.log" 2>&1
  echo "STRESS_EXIT=$?" >> "$LOG"
  run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" "2 8 8 16" DG_FP4_PUSH_DISPATCH=1
  echo ALL_STRESS_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  wait_idle; DG_FP4_PUSH_DISPATCH=1 LOG_TAG=_pushON${TAG}1 bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_PUSH_DISPATCH=0 LOG_TAG=_pushOFF${TAG}1 bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_PUSH_DISPATCH=1 LOG_TAG=_pushON${TAG}2 bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle; DG_FP4_PUSH_DISPATCH=1 LOG_TAG=_pushON${TAG}1 bash scripts/run_probe.sh qoq 8 > /dev/null 2>&1
  wait_idle; DG_FP4_PUSH_DISPATCH=0 LOG_TAG=_pushOFF${TAG}1 bash scripts/run_probe.sh qoq 8 > /dev/null 2>&1
  echo ALL_PERF_DONE >> "$LOG"
fi
echo ALL_DONE >> "$LOG"
