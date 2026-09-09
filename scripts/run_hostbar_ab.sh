#!/usr/bin/env bash
# Task A: official 24-report capture twice, customer method (DG_PROFILE_HOST_BARRIER=0)
# and with the all-rank host barrier before each launch (DG_PROFILE_HOST_BARRIER=1),
# then the last-3 summary + per-device reconcile of the Mega-only Fused reports.
# Usage (inside four_api_build): [MODES="off on"] bash scripts/run_hostbar_ab.sh /raid/kimi/results
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
RES=${1:-/raid/kimi/results}
export PYTHONPATH="$ROOT"
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
MARK="$RES/CAPTURE_RUNNING"
wait_idle() {
  local i
  for i in $(seq 1 300); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] &&
       ! ls "$RES"/OFFICIAL_*_RUNNING >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "GPUs still busy after 600 s" >&2; return 1
}
wait_idle || exit 1
echo "$$ $(date)" > "$MARK"
trap 'rm -f "$MARK"' EXIT
for mode in ${MODES:-off on}; do
  out="$RES/hostbar_$mode"
  bar=0; [ "$mode" = on ] && bar=1
  echo "=== capture host barrier=$mode -> $out ($(date))"
  DG_PROFILE_HOST_BARRIER=$bar OUT="$out" LOCK_SM_CLOCK_MHZ=1830 FORCE=1 \
    timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$out.log" 2>&1
  echo "capture $mode EXIT=$?"
  python3 scripts/summarize_four_api_h20_last3.py "$out" > "$out.last3.txt" 2>&1
  python3 scripts/reconcile_nsys_devices.py --last 8 "$out"/mega_fused_*.nsys-rep > "$out.reconcile.txt" 2>&1
done
echo HOSTBAR_AB_DONE
