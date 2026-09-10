#!/usr/bin/env bash
# Stream-K (DG_FP4_STREAMK) re-validation on the current tip (push dispatch default ON)
# and the customer-method A/B. Phases (MODE=corr|stress|ab|m8|all):
#   corr   : test_four_api_correctness mxfp4+qoq fused, T=2 8 8 16 (STREAMK=1), then the
#            owner-rank launches where stream-K actually activates: --tokens 1
#            --global-tokens 2 / 4 (STREAMK=1 and =0)
#   stress : 200-iteration graph-replay probe at M=2 and M=4 with STREAMK=1
#   ab     : official capture, TOKENS_LIST="2 4", knob 0/1, PASSES passes (run_fuse_customer_ab.sh
#            with KNOB=DG_FP4_STREAMK; result dirs $RES/sk_customer_k{0,1}_p{1,2})
#   m8     : one official capture TOKENS_LIST="8" with knob 1 ($RES/sk_customer_m8_k1_p1)
# GPU discipline as run_fuse_customer_ab.sh (60 s continuous idle, no foreign *_RUNNING marker,
# our marker CAPTURE_SK_RUNNING only while a GPU job runs).
# Usage (inside four_api_build): [MODE=all] bash scripts/run_sk_customer_ab.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-all}
RES=${RES:-/raid/kimi/results}
export PASSES=${PASSES:-2}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export CLOCK_LOCK_MODE=${CLOCK_LOCK_MODE:-verify}
export LOCK_SM_CLOCK_MHZ=${LOCK_SM_CLOCK_MHZ:-1830}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
TR=/usr/local/bin/torchrun
LOG=${LOG:-corr_sk_customer.log}
MARK="$RES/CAPTURE_SK_RUNNING"
mkdir -p "$RES"

other_markers() { ls "$RES"/*_RUNNING 2>/dev/null | grep -v "/CAPTURE_SK_RUNNING$"; }
gpus_free() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ -z "$(other_markers)" ]; }
wait_idle() {
  local i quiet=0
  for i in $(seq 1 10800); do
    if gpus_free; then quiet=$((quiet + 1)); [ "$quiet" -ge 6 ] && return 0; else quiet=0; fi
    sleep 10
  done
  echo "GPUs busy after 30 h" >&2; return 1
}
# run_gpu <label> <cmd...>: exclusive GPU job (marker held; redone if a foreign marker appeared)
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
corr() {  # corr <extra test args...> with env already set by caller via env(1)
  timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused "$@" >> "$LOG" 2>&1
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  echo "=== STREAMK corr $(git rev-parse --short HEAD) $(date)" >> "$LOG"
  for T in 2 8 8 16; do
    echo "--- STREAMK=1 T=$T (global $((T * 8)))" >> "$LOG"
    DG_FP4_STREAMK=1 run_gpu "corr T=$T" corr --tokens "$T"
  done
  for M in 2 4; do
    for K in 1 0; do
      echo "--- STREAMK=$K tokens=1 global-tokens=$M" >> "$LOG"
      DG_FP4_STREAMK=$K run_gpu "corr M=$M" corr --tokens 1 --global-tokens "$M"
    done
  done
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = stress ] || [ "$MODE" = all ]; then
  for M in 2 4; do
    echo "--- stress mxfp4 M=$M iters=200 STREAMK=1" >> "$LOG"
    DG_FP4_STREAMK=1 run_gpu "stress M=$M" timeout 900 "$TR" --standalone --nproc_per_node=8 \
      tests/profile_fused_phase_stamps.py --quant mxfp4 --global-tokens "$M" --iters 200 \
      > "probe_mxfp4_m${M}_sk_stress.log" 2>&1
    grep -E "PROBE_EXIT|Error|error|NaN|Traceback" "probe_mxfp4_m${M}_sk_stress.log" | head -5 >> "$LOG"
  done
  echo ALL_STRESS_DONE >> "$LOG"
fi

if [ "$MODE" = ab ] || [ "$MODE" = all ]; then
  KNOB=DG_FP4_STREAMK MARK=SK PREFIX=sk_customer MODE=customer TOKENS_LIST="2 4" SKIP_DONE=1 \
    bash scripts/run_fuse_customer_ab.sh >> "$LOG" 2>&1
  echo AB_DONE >> "$LOG"
fi

if [ "$MODE" = m8 ] || [ "$MODE" = all ]; then
  OUT="$RES/sk_customer_m8_k1_p1"
  if [ ! -f "$OUT/TIMELINE_LAST3.csv" ]; then
    echo "=== m8 knob=1 capture -> $OUT $(date)" >> "$LOG"
    DG_FP4_STREAMK=1 OUT="$OUT" RESUME=1 TOKENS_LIST="8" FORCE=1 run_gpu "m8 k1" \
      timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUT.log" 2>&1
    [ -f "$OUT/TIMELINE_LAST3.csv" ] && python3 scripts/reconcile_nsys_devices.py --last 3 \
      "$OUT"/*_fused_*.nsys-rep > "$OUT/reconcile_fused.txt" 2>&1
  fi
  echo M8_DONE >> "$LOG"
fi
echo ALL_DONE >> "$LOG"
