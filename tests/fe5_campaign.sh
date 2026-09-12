#!/bin/bash
# Plain customer-method campaign (8 GPUs, one build, one session): for M in 2 4 8 16 x mxfp4/qoq,
#   E2E scope (FE + Mega in one graph, nsys GPU0 median of the last 3 spans -> FE, in-graph Mega, E2E columns)
#   for 2 routing modes, interleaved per pass so drift hits every cell alike:
#     normal = the FE's real top-8 routing; balanced = DG_PROFILE_FORCE_BALANCED=1 (FE runs, one graph memcpy
#     overrides its routing with the Mega-only scope's balanced assignment before the Mega)
#   plus the Mega-only scope (no FE: identical for every routing mode) once per pass.
# Then scripts/summarize_four_api_h20_last3.py per capture directory.
# Usage (in four_api_build, repo root): bash tests/fe5_campaign.sh [passes=5] [outroot=/raid/kimi/results/fe5/cap]
cd "$(dirname "$0")/.."
PASSES=${1:-5}; ROOT_OUT=${2:-/raid/kimi/results/fe5/cap}; mkdir -p "$ROOT_OUT"
export DG_FE_SELECT_IN_MEGA=1 DG_PROFILE_HOST_BARRIER=${DG_PROFILE_HOST_BARRIER:-0}
echo "CAMPAIGN_START $(date +%T) $(git log --oneline -1) passes=$PASSES"
cap() {   # outdir scopes balanced
  local out=$1 scopes=$2
  if [ "$(ls "$out"/*.nsys-rep 2>/dev/null | wc -l)" -ge 8 ]; then echo "skip $out (complete)"; return; fi
  DG_PROFILE_FORCE_BALANCED=$3 OUT=$out SCOPES="$scopes" BACKENDS=fused QUANTS="mxfp4 qoq" TOKENS_LIST="2 4 8 16" FORCE=1 \
    timeout 2400 bash scripts/capture_four_api_h20_timelines.sh > "$out.log" 2>&1
  echo "cap $out rc=$? $(date +%T) reports=$(ls "$out"/*.nsys-rep 2>/dev/null | wc -l)"
}
for p in $(seq 1 "$PASSES"); do
  cap "$ROOT_OUT/lean_normal_p$p"   e2e  0
  cap "$ROOT_OUT/lean_balanced_p$p" e2e  1
  cap "$ROOT_OUT/megaonly_p$p"      mega 0
done
for d in "$ROOT_OUT"/*/; do
  echo "### $d"; SCOPES="$(basename "$d" | grep -q megaonly && echo mega || echo e2e)" BACKENDS=fused python3 scripts/summarize_four_api_h20_last3.py "$d" 2>&1 | grep -E "^\| (MXFP4|QOQ)" || echo "summarize failed for $d"
done
python3 tests/fe5_summarize_campaign.py "$ROOT_OUT"
echo "CAMPAIGN_DONE $(date +%T)"
