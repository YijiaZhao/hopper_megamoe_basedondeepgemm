#!/usr/bin/env bash
# Fable tiny-M frontend: DG_FE_ROUTER_L2_PERSIST (router weights pinned in L2) and
# DG_FE_PDL (programmatic dependent launch FE -> fused Mega) A/B. Phases (MODE=ident|bench|corr|all):
#   ident : tests/test_frontend_tinym.py --l2-persist {1,2}: tiny-M + knob vs legacy, bit-identical
#   bench : tests/bench_frontend_tinym.py (8 ranks, DG_FE_STAMPS=1) for every state in STATES
#           ("l2p:pdl" pairs) x quant x M in TOKENS_LIST: FE / Mega / FE+Mega CUDA-event timing +
#           stamps (eager after flush, and inside the FE+Mega graph)
#   corr  : test_four_api_correctness mxfp4+qoq fused, T=2 8 8 16 with DG_FE_PDL=1 (+ L2P=CORR_L2P)
# GPU discipline as run_fe_tinym.sh; our marker is CAPTURE_FE2_RUNNING.
# Usage (inside four_api_build): [MODE=all] [TOKENS_LIST="2 8"] [STATES="0:0 1:0 2:0 0:1 1:1"] bash scripts/run_fe_l2pdl.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-all}
RES=${RES:-/raid/kimi/results}
TOKENS_LIST=${TOKENS_LIST:-"2 8"}
STATES=${STATES:-"0:0 1:0 2:0 0:1 1:1"}
CORR_L2P=${CORR_L2P:-1}
ITERS=${ITERS:-100}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_FE_TINYM=1
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
TR=/usr/local/bin/torchrun
LOG=${LOG:-$RES/fe2/fe_l2pdl.log}
MARK="$RES/CAPTURE_FE2_RUNNING"
mkdir -p "$RES/fe2"

other_markers() { ls "$RES"/*_RUNNING 2>/dev/null | grep -v "/CAPTURE_FE2_RUNNING$"; }
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

echo "=== FE L2P/PDL $MODE $(git rev-parse --short HEAD) $(date)" >> "$LOG"
if [ "$MODE" = ident ] || [ "$MODE" = all ]; then
  for P in 1 2; do
    echo "--- ident l2_persist=$P" >> "$LOG"
    run_gpu "ident P=$P" env CUDA_VISIBLE_DEVICES=7 timeout 600 python3 tests/test_frontend_tinym.py --seeds 100 --l2-persist "$P" >> "$LOG" 2>&1
  done
fi
if [ "$MODE" = bench ] || [ "$MODE" = all ]; then
  for M in $TOKENS_LIST; do
    for Q in mxfp4 qoq; do
      for S in $STATES; do
        L2P=${S%%:*}; PDL=${S##*:}
        echo "--- bench M=$M quant=$Q DG_FE_ROUTER_L2_PERSIST=$L2P DG_FE_PDL=$PDL" >> "$LOG"
        DG_FE_ROUTER_L2_PERSIST=$L2P DG_FE_ROUTER_L2_PERSIST_VERBOSE=1 DG_FE_PDL=$PDL DG_FE_STAMPS=1 \
          run_gpu "bench M=$M $Q l2p$L2P pdl$PDL" timeout 600 "$TR" --standalone --nproc_per_node=8 \
          tests/bench_frontend_tinym.py --quant "$Q" --global-tokens "$M" --iters "$ITERS" >> "$LOG" 2>&1
      done
    done
  done
fi
if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  for T in 2 8 8 16; do
    echo "--- corr DG_FE_PDL=1 DG_FE_ROUTER_L2_PERSIST=$CORR_L2P T=$T (global $((T * 8)))" >> "$LOG"
    DG_FE_PDL=1 DG_FE_ROUTER_L2_PERSIST=$CORR_L2P run_gpu "corr T=$T" timeout 600 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1
  done
fi
echo "ALL_DONE $MODE $(date)" >> "$LOG"
