#!/usr/bin/env bash
# Fable frontend tiny-M (DG_FE_TINYM) validation + direct timing. Phases (MODE=ident|bench|corr|all):
#   ident : tests/test_frontend_tinym.py (single GPU, last visible device): 100 seeds x
#           rows {1,2,4,8,16} x {mxfp4,qoq}, knob 0 vs 1 bit-identical
#   bench : tests/bench_frontend_tinym.py (8 ranks) for knob 0/1 x quant x M in TOKENS_LIST,
#           100 iterations, FE / Mega / FE+Mega direct CUDA-event timing + DG_FE_STAMPS attribution
#   corr  : test_four_api_correctness mxfp4+qoq fused, T=2 8 8 16 with DG_FE_TINYM=1
# GPU discipline as run_sk_customer_ab.sh (60 s continuous idle, no foreign *_RUNNING marker,
# our marker CAPTURE_FE_RUNNING only while a GPU job runs).
# Usage (inside four_api_build): [MODE=all] [TOKENS_LIST="2 4 8 16"] bash scripts/run_fe_tinym.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-all}
RES=${RES:-/raid/kimi/results}
TOKENS_LIST=${TOKENS_LIST:-"2 4 8 16"}
ITERS=${ITERS:-100}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
TR=/usr/local/bin/torchrun
LOG=${LOG:-$RES/fe/fe_tinym.log}
MARK="$RES/CAPTURE_FE_RUNNING"
mkdir -p "$RES/fe"

other_markers() { ls "$RES"/*_RUNNING 2>/dev/null | grep -v "/CAPTURE_FE_RUNNING$"; }
gpus_free() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ -z "$(other_markers)" ]; }
wait_idle() {
  local i quiet=0
  for i in $(seq 1 10800); do
    if gpus_free; then quiet=$((quiet + 1)); [ "$quiet" -ge 6 ] && return 0; else quiet=0; fi
    sleep 10
  done
  echo "GPUs busy after 30 h" >&2; return 1
}
run_gpu() {
  local label=$1; shift
  local attempt
  for attempt in 1 2 3; do
    wait_idle || return 1
    echo "$$ $(date) $label" > "$MARK"
    "$@"; local rc=$?
    local foreign; foreign=$(other_markers)
    rm -f "$MARK"
    if [ -n "$foreign" ]; then echo "--- $label overlapped foreign marker ($foreign): redo" >> "$LOG"; continue; fi
    echo "EXIT=$rc" >> "$LOG"; return $rc
  done
  return 1
}

echo "=== FE tiny-M $MODE $(git rev-parse --short HEAD) $(date)" >> "$LOG"
if [ "$MODE" = ident ] || [ "$MODE" = all ]; then
  echo "--- ident" >> "$LOG"
  run_gpu ident env CUDA_VISIBLE_DEVICES=7 timeout 900 python3 tests/test_frontend_tinym.py --seeds 100 >> "$LOG" 2>&1
fi
if [ "$MODE" = bench ] || [ "$MODE" = all ]; then
  for M in $TOKENS_LIST; do
    for Q in mxfp4 qoq; do
      for K in 0 1; do
        echo "--- bench M=$M quant=$Q DG_FE_TINYM=$K" >> "$LOG"
        DG_FE_TINYM=$K DG_FE_STAMPS=1 run_gpu "bench M=$M $Q k$K" timeout 900 "$TR" --standalone --nproc_per_node=8 \
          tests/bench_frontend_tinym.py --quant "$Q" --global-tokens "$M" --iters "$ITERS" >> "$LOG" 2>&1
      done
    done
  done
fi
if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  for T in 2 8 8 16; do
    echo "--- corr DG_FE_TINYM=1 T=$T (global $((T * 8)))" >> "$LOG"
    DG_FE_TINYM=1 run_gpu "corr T=$T" timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1
  done
fi
echo "ALL_DONE $MODE $(date)" >> "$LOG"
