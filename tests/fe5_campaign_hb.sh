#!/bin/bash
# Round-5 supplementary campaign WITH the host barrier before every graph replay (DG_PROFILE_HOST_BARRIER=1, removes
# most of the inter-rank launch skew): E2E, LEAN=1 only, normal and forced-balanced routing, both quants, M 2 4 8 16,
# 3 interleaved passes + one Mega-only pass. Separate output root so it is never mixed with the plain customer method.
# Usage (in four_api_build / fe5c_build, repo root): bash tests/fe5_campaign_hb.sh [passes=3] [outroot=/raid/kimi/results/fe5/cap_hb]
cd "$(dirname "$0")/.."
PASSES=${1:-3}; ROOT_OUT=${2:-/raid/kimi/results/fe5/cap_hb}; mkdir -p "$ROOT_OUT"
export DG_FE_SELECT_IN_MEGA=1 DG_PROFILE_HOST_BARRIER=1 DG_FE_CC_LEAN=1
echo "CAMPAIGN_HB_START $(date +%T) $(git log --oneline -1) passes=$PASSES"
cap() {   # outdir scopes balanced
  local out=$1 scopes=$2
  DG_PROFILE_FORCE_BALANCED=$3 OUT=$out SCOPES="$scopes" BACKENDS=fused QUANTS="mxfp4 qoq" TOKENS_LIST="2 4 8 16" FORCE=1 \
    timeout 2400 bash scripts/capture_four_api_h20_timelines.sh > "$out.log" 2>&1
  echo "cap $out rc=$? $(date +%T) reports=$(ls "$out"/*.nsys-rep 2>/dev/null | wc -l)"
}
for p in $(seq 1 "$PASSES"); do
  cap "$ROOT_OUT/lean_normal_p$p"   e2e 0
  cap "$ROOT_OUT/lean_balanced_p$p" e2e 1
done
cap "$ROOT_OUT/megaonly_p1" mega 0
for d in "$ROOT_OUT"/*/; do
  echo "### $d"; SCOPES="$(basename "$d" | grep -q megaonly && echo mega || echo e2e)" BACKENDS=fused python3 scripts/summarize_four_api_h20_last3.py "$d" 2>&1 | grep -E "^\| (MXFP4|QOQ)" || echo "summarize failed for $d"
done
python3 tests/fe5_summarize_campaign.py "$ROOT_OUT"
echo "CAMPAIGN_HB_DONE $(date +%T)"
