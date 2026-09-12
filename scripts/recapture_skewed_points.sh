#!/usr/bin/env bash
# Re-capture the fused points of a customer-method matrix whose last-3 start skew exceeds
# SKEW_MAX (default 20 us, reconcile_nsys_devices.py): delete those reports, re-run the
# capture with RESUME=1 (only the missing cases are captured, each behind the per-case idle
# checks), re-reconcile; up to ROUNDS rounds. Then refresh the summaries.
# Usage (inside four_api_build): bash scripts/recapture_skewed_points.sh OUTDIR
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
OUT=$1
SKEW_MAX=${SKEW_MAX:-20}; ROUNDS=${ROUNDS:-4}
export PYTHONPATH="$ROOT"
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export TOKENS_LIST=${TOKENS_LIST:-"2 8 16"}
export CLOCK_LOCK_MODE=${CLOCK_LOCK_MODE:-verify}
RES=${RES:-/raid/kimi/results}
other_markers() { ls "$RES"/*_RUNNING 2>/dev/null | grep -v "/CAPTURE_FUSE_RUNNING$"; }
gpus_free() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ -z "$(other_markers)" ]; }
wait_idle() {
  local i quiet=0
  for i in $(seq 1 10800); do
    if gpus_free; then quiet=$((quiet + 1)); [ "$quiet" -ge 6 ] && return 0; else quiet=0; fi
    sleep 10
  done
  return 1
}
skewed() {  # fused reports with start skew > SKEW_MAX
  python3 scripts/reconcile_nsys_devices.py --last 3 "$OUT"/*_fused_*.nsys-rep 2>/dev/null |
    awk -v mx="$SKEW_MAX" '/^## /{n=$2; sub(":$","",n)} /official window/{if ($13 + 0 > mx) print n}'
}
for round in $(seq 1 "$ROUNDS"); do
  bad=$(skewed)
  [ -z "$bad" ] && break
  echo "round $round: re-capturing (skew > $SKEW_MAX us): $bad"
  for f in $bad; do rm -f "$OUT/$f"; done
  wait_idle || exit 1
  OUT="$OUT" RESUME=1 timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUT.recapture$round.log" 2>&1
  echo "RECAPTURE_EXIT=$? $(date)"
done
python3 scripts/reconcile_nsys_devices.py --last 3 "$OUT"/*_fused_*.nsys-rep > "$OUT/reconcile_fused.txt" 2>&1
python3 scripts/summarize_four_api_h20_last3.py "$OUT" > /dev/null 2>&1
echo "remaining skewed: $(skewed | tr '\n' ' ')"
grep fused "$OUT/TIMELINE_LAST3.csv" | cut -d, -f1-3,6-8
awk '/^## /{n=$2} /official window/{print n, "gpu0", $6, "minDev", $8, "skew", $13}' "$OUT/reconcile_fused.txt"
echo RECAPTURE_DONE
