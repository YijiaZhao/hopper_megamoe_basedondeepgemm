#!/usr/bin/env bash
# DG_FP4_FUSE_L1L2 perf A/B, customer method first: the official capture
# (scripts/capture_four_api_h20_timelines.sh, full 2 scopes x 2 backends x 2 quants x
# TOKENS_LIST matrix, GPU0 median of the last 3 spans via summarize_four_api_h20_last3.py),
# knob 0 / 1, PASSES passes (pass 2 in reverse knob order), each into its own OUT dir under
# $RES; then reconcile_nsys_devices.py on the fused reports (start skew per point: > 20 us
# means re-capture that point). Finally the skew-free min-over-devices A/B
# (run_knob_nsys_ab.sh, mega fused only) as the footnote.
# Usage (inside four_api_build, GPUs idle, clocks locked): [MODE=customer|skewfree|all] bash scripts/run_fuse_customer_ab.sh [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TAG=${1:-}
RES=${RES:-/raid/kimi/results}
PASSES=${PASSES:-2}
export TOKENS_LIST=${TOKENS_LIST:-"2 8 16"}
# verify (default): the H20 boxes are locked at 1830 MHz on the host; the container cannot -lgc
export CLOCK_LOCK_MODE=${CLOCK_LOCK_MODE:-verify}
MODE=${MODE:-all}  # customer | skewfree | all
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$ROOT"
mkdir -p "$RES"
echo "$$ $(date)" > "$RES/CAPTURE_FUSE_RUNNING"
trap 'rm -f "$RES/CAPTURE_FUSE_RUNNING"' EXIT
# Strict coordination: no other *_RUNNING marker of any kind (ours excepted) and no
# compute process on any GPU, re-checked 10 s later, before every matrix; a matrix whose
# capture exits non-zero (the capture script's own per-case idle checks fail when another
# job appears mid-matrix) is discarded and re-captured (up to 3 attempts).
other_markers() {
  ls "$RES"/*_RUNNING 2>/dev/null | grep -v "/CAPTURE_FUSE_RUNNING$"
}
gpus_free() {
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ -z "$(other_markers)" ]
}
# Start only after 60 s of continuous emptiness (a restarting server leaves short gaps)
wait_idle() {
  local i quiet=0
  for i in $(seq 1 10800); do
    if gpus_free; then quiet=$((quiet + 1)); [ "$quiet" -ge 6 ] && return 0; else quiet=0; fi
    sleep 10
  done
  echo "GPUs busy after 30 h" >&2; return 1
}
echo "build $(git rev-parse --short HEAD) $(date)"
[ "$MODE" = skewfree ] || for pass in $(seq 1 "$PASSES"); do
  knobs="0 1"; [ $((pass % 2)) -eq 0 ] && knobs="1 0"
  for knob in $knobs; do
    OUT="$RES/fuse_customer${TAG}_k${knob}_p${pass}"
    if [ "${SKIP_DONE:-0}" = 1 ] && [ -f "$OUT/TIMELINE_LAST3.csv" ]; then
      echo "=== customer capture knob=$knob pass=$pass -> $OUT already complete, kept"
      cat "$OUT/TIMELINE_LAST3.csv"; continue
    fi
    # Resumable: every kept report passed the capture script's per-case idle checks
    # (a case interrupted by another job is deleted there); retry until the matrix is
    # complete or the deadline passes.
    attempt=0; deadline=$(( $(date +%s) + ${DEADLINE_H:-12} * 3600 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      attempt=$((attempt + 1))
      wait_idle || exit 1
      echo "=== customer capture knob=$knob pass=$pass attempt=$attempt -> $OUT $(date)"
      OUT="$OUT" RESUME=1 DG_FP4_FUSE_L1L2=$knob timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUT.log" 2>&1
      rc=$?
      echo "CAPTURE_EXIT=$rc $(date) (clean reports so far: $(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l))"
      [ "$rc" -eq 0 ] && break
    done
    [ -f "$OUT/TIMELINE_LAST3.csv" ] || continue
    python3 scripts/reconcile_nsys_devices.py --last 3 "$OUT"/*_fused_*.nsys-rep > "$OUT/reconcile_fused.txt" 2>&1
    grep -E "^##|last3|skew" "$OUT/reconcile_fused.txt" | head -40
    cat "$OUT/TIMELINE_LAST3.csv" 2>/dev/null
  done
done
echo CUSTOMER_AB_DONE
[ "$MODE" = customer ] && exit 0
# Footnote: skew-free (host barrier) min-over-devices, mega fused only
wait_idle || exit 1
KNOB=DG_FP4_FUSE_L1L2 OFF=0 ON=1 PASSES="$PASSES" bash scripts/run_knob_nsys_ab.sh \
  "$RES/fuse_l1l2_skewfree$TAG" mxfp4:8 mxfp4:2 mxfp4:16 qoq:8 qoq:16
echo SKEWFREE_AB_DONE
