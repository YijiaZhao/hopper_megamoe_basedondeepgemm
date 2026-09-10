#!/usr/bin/env bash
# DG_FP4_FUSE_L1L2 perf A/B, customer method first: the official capture
# (scripts/capture_four_api_h20_timelines.sh, full 2 scopes x 2 backends x 2 quants x
# TOKENS_LIST matrix, GPU0 median of the last 3 spans via summarize_four_api_h20_last3.py),
# knob 0 / 1, PASSES passes (pass 2 in reverse knob order), each into its own OUT dir under
# $RES; then reconcile_nsys_devices.py on the fused reports (start skew per point: > 20 us
# means re-capture that point). Finally the skew-free min-over-devices A/B
# (run_knob_nsys_ab.sh, mega fused only) as the footnote.
# Usage (inside four_api_build, GPUs idle, clocks lockable): bash scripts/run_fuse_customer_ab.sh [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TAG=${1:-}
RES=${RES:-/raid/kimi/results}
PASSES=${PASSES:-2}
export TOKENS_LIST=${TOKENS_LIST:-"2 8 16"}
export CLOCK_LOCK_MODE=${CLOCK_LOCK_MODE:-set}
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$ROOT"
mkdir -p "$RES"
echo "$$ $(date)" > "$RES/CAPTURE_FUSE_RUNNING"
trap 'rm -f "$RES/CAPTURE_FUSE_RUNNING"' EXIT
wait_idle() {
  local i
  for i in $(seq 1 1800); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] &&
       ! ls "$RES"/OFFICIAL_*_RUNNING "$RES"/CAPTURE_RUNNING "$RES"/CAPTURE_M*_RUNNING >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "GPUs busy after 1 h" >&2; return 1
}
echo "build $(git rev-parse --short HEAD) $(date)"
for pass in $(seq 1 "$PASSES"); do
  knobs="0 1"; [ $((pass % 2)) -eq 0 ] && knobs="1 0"
  for knob in $knobs; do
    OUT="$RES/fuse_customer${TAG}_k${knob}_p${pass}"
    wait_idle || exit 1
    echo "=== customer capture knob=$knob pass=$pass -> $OUT $(date)"
    OUT="$OUT" FORCE=1 DG_FP4_FUSE_L1L2=$knob timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUT.log" 2>&1
    echo "CAPTURE_EXIT=$?"
    python3 scripts/reconcile_nsys_devices.py --last 3 "$OUT"/*_fused_*.nsys-rep > "$OUT/reconcile_fused.txt" 2>&1
    grep -E "^##|last3|skew" "$OUT/reconcile_fused.txt" | head -40
    cat "$OUT/TIMELINE_LAST3.csv" 2>/dev/null
  done
done
echo CUSTOMER_AB_DONE
# Footnote: skew-free (host barrier) min-over-devices, mega fused only
wait_idle || exit 1
KNOB=DG_FP4_FUSE_L1L2 OFF=0 ON=1 PASSES="$PASSES" bash scripts/run_knob_nsys_ab.sh \
  "$RES/fuse_l1l2_skewfree$TAG" mxfp4:8 mxfp4:2 mxfp4:16 qoq:8 qoq:16
echo SKEWFREE_AB_DONE
