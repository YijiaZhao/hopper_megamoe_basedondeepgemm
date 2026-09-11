#!/usr/bin/env bash
# H20 M=16 task-shape A/B runner (host 10.6.131.8, container nvfp4_timeline).
#   corr    <log> [ENV=..]                 : mxfp4+qoq fused correctness T=2 8 8 16 (+ mxfp4 128 512 with BIG=1)
#   capture <outdir> <passes> <tokens> [ENV=..] : N independent official captures -> <outdir>/p<i>, then
#                                           scripts/summarize_m16_passes.py over the passes
# Every GPU run waits for idle GPUs first (shared box), is wrapped in timeout 900,
# and never touches the clocks (verify mode).
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
TR=$(command -v torchrun)
MODE=${1:-}; shift || true

# Shared-box discipline: MARKER (e.g. /raid/kimi/results/CAPTURE_M16_RUNNING) is created while
# this runner holds the GPUs; wait_idle also waits while any FOREIGN *_RUNNING marker exists
# in the marker directory. Other containers' processes are invisible here, so the GPU gate
# is memory.used (<= 64 MiB on all 8 GPUs) + no visible compute app.
MARKER=${MARKER:-}
foreign_marker() {
  [ -n "$MARKER" ] || return 1
  local f
  for f in "$(dirname "$MARKER")"/*_RUNNING; do
    [ -e "$f" ] || continue
    [ "$f" = "$MARKER" ] || return 0
  done
  return 1
}
hold_marker() { [ -n "$MARKER" ] && echo "$$ $(date -u +%FT%TZ) $*" > "$MARKER"; }
drop_marker() { [ -n "$MARKER" ] && rm -f "$MARKER"; }
trap drop_marker EXIT
wait_idle() {
  for _ in $(seq 1 4320); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 64 {n++} END {print n+0}')
    if [ "$busy" = 0 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && ! foreign_marker; then
      hold_marker "$MODE"; return 0
    fi
    drop_marker; sleep 10
  done
  echo "GPUs busy for 12 h, giving up" >&2; return 1
}

case "$MODE" in
  corr)
    LOG=$1; shift
    : > "$LOG"
    rc=0
    for T in 2 8 8 16; do
      echo "--- mxfp4+qoq fused T=$T ($*)" >> "$LOG"
      wait_idle
      env "$@" timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
        --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1 || rc=1
    done
    if [ "${BIG:-0}" = 1 ]; then
      for T in 128 512; do
        echo "--- mxfp4 fused T=$T ($*)" >> "$LOG"
        wait_idle
        env "$@" timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
          --apis mxfp4_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1 || rc=1
      done
    fi
    echo "CORR_RC=$rc" >> "$LOG"
    drop_marker
    ;;
  capture)
    OUTBASE=$1; PASSES=$2; TOK=$3; shift 3
    mkdir -p "$OUTBASE"
    LOG="$OUTBASE/capture.log"
    rc=0
    for p in $(seq 1 "$PASSES"); do
      echo "=== pass $p ($*) tokens=$TOK $(date)" >> "$LOG"
      wait_idle
      env "$@" TOKENS_LIST="$TOK" OUT="$OUTBASE/p$p" LOCK_SM_CLOCK_MHZ=1830 FORCE=1 \
        timeout 3600 bash scripts/capture_four_api_h20_timelines.sh >> "$LOG" 2>&1 || { echo "pass $p rc=$?" >> "$LOG"; rc=1; }
    done
    python3 scripts/summarize_m16_passes.py "$OUTBASE"/p* > "$OUTBASE/SUMMARY.md" 2>> "$LOG"
    echo "CAPTURE_RC=$rc" >> "$LOG"
    drop_marker
    ;;
  *) echo "usage: $0 corr <log> [ENV..] | capture <outdir> <passes> <tokens_list> [ENV..]" >&2; exit 2 ;;
esac
