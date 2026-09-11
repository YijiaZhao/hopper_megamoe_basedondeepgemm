#!/usr/bin/env bash
# Fable frontend full-K grid (DG_FE_TINYM_GRID=auto vs 96) validation + timing. Phases
# (MODE=ident|bench|corr|ncu|capture|all):
#   ident   : tests/test_frontend_fe78.py (single GPU, last visible device): top-8 index sets
#             + 1e-6 weights, 96 vs auto, 40 seeds x rows {1,2,8,16} x {mxfp4,qoq}
#   bench   : tests/bench_frontend_tinym.py (8 ranks, DG_FE_STAMPS=1) for grid 96 / auto x
#             quant x M in TOKENS_LIST (8 -> 1 row/rank, 16 -> 2 rows/rank)
#   corr    : test_four_api_correctness mxfp4+qoq fused, T=2 8 8 16 with the default grid (auto)
#   ncu     : one --set full report of the full-K kernel (rows 1, mxfp4) into $RES/ncu_fe + SUMMARY.md
#   capture : customer-method nsys (SCOPES=e2e BACKENDS=fused TOKENS_LIST="2 8") per grid knob
# GPU discipline: 60 s continuous idle, no foreign *_RUNNING marker, our marker
# CAPTURE_FE78_RUNNING only while a GPU job runs.
# Usage (inside four_api_build, repo root): [MODE=all] [TOKENS_LIST="8 16"] bash scripts/run_fe78.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-all}
RES=${RES:-/raid/kimi/results}
TOKENS_LIST=${TOKENS_LIST:-"8 16"}
ITERS=${ITERS:-100}
GRIDS=${GRIDS:-"96 auto"}
MMAS=${MMAS:-"wmma"}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_FE_TINYM=1
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
TR=/usr/local/bin/torchrun
LOG=${LOG:-$RES/fe/fe78.log}
MARK="$RES/CAPTURE_FE78_RUNNING"
mkdir -p "$RES/fe" "$RES/ncu_fe"

other_markers() { ls "$RES"/*_RUNNING 2>/dev/null | grep -v "/CAPTURE_FE78_RUNNING$"; }
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

echo "=== FE78 $MODE $(git rev-parse --short HEAD) $(date)" >> "$LOG"
if [ "$MODE" = ident ] || [ "$MODE" = all ]; then
  echo "--- ident (96 vs auto)" >> "$LOG"
  for MM in $MMAS; do
    run_gpu ident env CUDA_VISIBLE_DEVICES=7 timeout 900 python3 tests/test_frontend_fe78.py --seeds 40 --mma "$MM" >> "$LOG" 2>&1
  done
fi
if [ "$MODE" = bench ] || [ "$MODE" = all ]; then
  for M in $TOKENS_LIST; do
    for Q in mxfp4 qoq; do
      for G in $GRIDS; do for MM in $MMAS; do
        echo "--- bench M=$M quant=$Q DG_FE_TINYM_GRID=$G DG_FE_TINYM_MMA=$MM" >> "$LOG"
        DG_FE_TINYM_GRID=$G DG_FE_TINYM_MMA=$MM DG_FE_STAMPS=1 run_gpu "bench M=$M $Q g$G $MM" timeout 900 "$TR" --standalone --nproc_per_node=8 \
          tests/bench_frontend_tinym.py --quant "$Q" --global-tokens "$M" --iters "$ITERS" >> "$LOG" 2>&1
      done; done
    done
  done
fi
if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  for T in 2 8 8 16; do
    echo "--- corr DG_FE_TINYM_GRID=auto T=$T (global $((T * 8)))" >> "$LOG"
    DG_FE_TINYM_GRID=auto run_gpu "corr T=$T" timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1
  done
fi
if [ "$MODE" = ncu ] || [ "$MODE" = all ]; then
  echo "--- ncu full-K rows=1 mxfp4 (fe_fullkauto_M8_mxfp4)" >> "$LOG"
  OUT="$RES/ncu_fe"; TAG=${NCU_TAG:-fe_fullkauto_M8_mxfp4}
  run_gpu ncu env CUDA_VISIBLE_DEVICES=7 DG_FE_TINYM_GRID=auto timeout 900 ncu --target-processes application-only \
    --kernel-name regex:router_quant_topk_kernel --launch-skip 5 --launch-count 1 \
    --set full --import-source no --clock-control none -f -o "$OUT/$TAG" \
    python3 tests/ncu_frontend_tinym.py --quant mxfp4 --rows 1 --tinym 1 --grid auto --warmup 5 > "$OUT/$TAG.run.log" 2>&1
  if [ -f "$OUT/$TAG.ncu-rep" ]; then
    ncu --import "$OUT/$TAG.ncu-rep" --page details > "$OUT/$TAG.details.txt" 2>&1
    ncu --import "$OUT/$TAG.ncu-rep" --page details --csv > "$OUT/$TAG.details.csv" 2>&1
    ncu --import "$OUT/$TAG.ncu-rep" --page raw --csv > "$OUT/$TAG.raw.csv" 2>&1
    python3 tests/ncu_fe_summary.py "$OUT" > "$OUT/SUMMARY.md"
  fi
  tail -3 "$OUT/$TAG.run.log" >> "$LOG"
fi
if [ "$MODE" = capture ] || [ "$MODE" = all ]; then
  for G in $GRIDS; do
    OUTDIR="$RES/fe78_nsys_grid$G"
    echo "--- capture DG_FE_TINYM_GRID=$G -> $OUTDIR" >> "$LOG"
    DG_FE_TINYM_GRID=$G TOKENS_LIST="2 8" OUT="$OUTDIR" LOCK_SM_CLOCK_MHZ=1830 FORCE=1 SCOPES=e2e BACKENDS=fused \
      run_gpu "capture g$G" timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUTDIR.log" 2>&1
    grep -A 20 "TIMELINE_TABLE\|frontend_us" "$OUTDIR/TIMELINE_TABLE.md" 2>/dev/null | head -30 >> "$LOG"
  done
fi
echo "ALL_DONE $MODE $(date)" >> "$LOG"
