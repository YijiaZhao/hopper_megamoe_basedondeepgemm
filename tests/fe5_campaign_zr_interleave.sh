#!/bin/bash
# Re-measurement campaign, zero-row knob OFF/ON interleaved (README method, tests/fe5_campaign_streamed.sh extended).
# Per pass: E2E normal OFF, E2E normal ON, E2E balanced OFF, E2E balanced ON, Mega-only (knob-independent).
# Usage (inside the container, repo root): [SKIP_MEGA=1] bash fe5_campaign_zr_interleave.sh [passes=5] [outroot=/raid/kimi/results/remeasure/cap] [first_pass=1]
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$REPO" || exit 1
PASSES=${1:-5}; ROOT_OUT=${2:-/raid/kimi/results/remeasure/cap}; FIRST=${3:-1}
mkdir -p "$ROOT_OUT/off" "$ROOT_OUT/on" "$ROOT_OUT/mega"
export DG_FE_SELECT_IN_MEGA=1 DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30
echo "CAMPAIGN_ZR_START $(date -u +%FT%TZ) $(git log --oneline -1) passes=$PASSES first=$FIRST"
cap() {   # outdir scopes balanced zero_row
  local out=$1 scopes=$2
  DG_FE_ZERO_ROW_UNROUTED=$4 DG_PROFILE_FORCE_BALANCED=$3 OUT=$out SCOPES="$scopes" BACKENDS=fused QUANTS="mxfp4 qoq" TOKENS_LIST="2 4 8 16" FORCE=1 \
    timeout 2400 bash scripts/capture_four_api_h20_timelines.sh > "$out.log" 2>&1
  echo "cap $out rc=$? $(date -u +%T) reports=$(ls "$out"/*.nsys-rep 2>/dev/null | wc -l)"
}
for p in $(seq "$FIRST" $((FIRST + PASSES - 1))); do
  cap "$ROOT_OUT/off/zr0_normal_p$p"    e2e  0 0
  cap "$ROOT_OUT/on/zr1_normal_p$p"     e2e  0 1
  cap "$ROOT_OUT/off/zr0_balanced_p$p"  e2e  1 0
  cap "$ROOT_OUT/on/zr1_balanced_p$p"   e2e  1 1
  [ "${SKIP_MEGA:-0}" = 1 ] || cap "$ROOT_OUT/mega/megaonly_p$p" mega 0 1
done
for sub in off on mega; do echo "### summary $sub"; python3 tests/fe5_summarize_campaign.py "$ROOT_OUT/$sub"; done
echo "CAMPAIGN_ZR_DONE $(date -u +%FT%TZ)"
