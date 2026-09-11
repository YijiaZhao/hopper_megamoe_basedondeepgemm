#!/usr/bin/env bash
# Validate the dynamic combine token claim (kCombineDynamic, DG_FP4_COMBINE_DYNAMIC):
# correctness (mxfp4 + qoq T = 2 8 8 16 twice back-to-back, mxfp4 T = 128 512), a
# 200-iteration graph-replay stress probe at M = 8, phase-stamp probe ON vs OFF
# (slots 5/6/7), then the skew-free official-method knob A/B (run_knob_nsys_ab.sh).
# Waits for idle GPUs before every GPU run; perf also waits for *_RUNNING markers.
# Usage (inside four_api_build): bash scripts/run_combine_dynamic_validate.sh [corr|stress|probe|perf|all] [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}
TAG=${2:-}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export TORCH_CUDA_ARCH_LIST=9.0a
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
TR=/usr/local/bin/torchrun
RES=/raid/kimi/results
LOG=cmb_validate$TAG.log

wait_idle() {
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    m=$(ps -C nvcc -o stat= 2>/dev/null | grep -v '^Z' | grep -c .)
    [ "$n" = 0 ] && [ "$m" = 0 ] && break
    sleep 5
  done
}
wait_idle_perf() {
  while true; do
    wait_idle
    ls "$RES"/*_RUNNING >/dev/null 2>&1 || break
    sleep 10
  done
}

run_corr() {  # $1 = api list, $2 = token list (one launch per T), $3.. = env assignments
  local apis=$1 toks=$2 T; shift 2
  for T in $toks; do
    wait_idle
    echo "--- apis=[$apis] tokens=$T ($*)" >> "$LOG"
    env "$@" timeout 600 $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis $apis --tokens $T >> "$LOG" 2>&1
    echo "EXIT=$?" >> "$LOG"
  done
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  : > "$LOG"
  run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" "2 8 8 16" DG_FP4_COMBINE_DYNAMIC=1 DG_FP4_SPIN_TIMEOUT=1
  run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" "2 8 8 16" DG_FP4_COMBINE_DYNAMIC=1
  run_corr "mxfp4_mega_moe_fused" "128 512" DG_FP4_COMBINE_DYNAMIC=1
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = stress ] || [ "$MODE" = all ]; then
  wait_idle
  echo "--- stress mxfp4 M=8 iters=200 DYNAMIC=1" >> "$LOG"
  DG_FP4_COMBINE_DYNAMIC=1 timeout 600 $TR --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py \
    --quant mxfp4 --global-tokens 8 --iters 200 > "probe_mxfp4_m8_cmb_stress$TAG.log" 2>&1
  echo "STRESS_EXIT=$?" >> "$LOG"
  run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" "2 8 8 16" DG_FP4_COMBINE_DYNAMIC=1
  echo ALL_STRESS_DONE >> "$LOG"
fi

if [ "$MODE" = probe ] || [ "$MODE" = all ]; then
  wait_idle_perf; DG_FP4_COMBINE_DYNAMIC=1 LOG_TAG=_cmbON${TAG}1 bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle_perf; DG_FP4_COMBINE_DYNAMIC=0 LOG_TAG=_cmbOFF${TAG}1 bash scripts/run_probe.sh mxfp4 2 8 16 > /dev/null 2>&1
  wait_idle_perf; DG_FP4_COMBINE_DYNAMIC=1 LOG_TAG=_cmbON${TAG}1 bash scripts/run_probe.sh qoq 8 > /dev/null 2>&1
  wait_idle_perf; DG_FP4_COMBINE_DYNAMIC=0 LOG_TAG=_cmbOFF${TAG}1 bash scripts/run_probe.sh qoq 8 > /dev/null 2>&1
  echo ALL_PROBE_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  wait_idle_perf
  KNOB=DG_FP4_COMBINE_DYNAMIC OFF=0 ON=1 PASSES=2 bash scripts/run_knob_nsys_ab.sh "$RES/cmb_ab$TAG" \
    mxfp4:8 mxfp4:2 mxfp4:16 qoq:8 > "cmb_ab$TAG.log" 2>&1
  echo "PERF_EXIT=$?" >> "$LOG"
  echo ALL_PERF_DONE >> "$LOG"
fi
echo ALL_DONE >> "$LOG"
