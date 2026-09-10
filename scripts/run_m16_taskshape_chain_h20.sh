#!/usr/bin/env bash
# H20 .8 M=16 task-shape measurement chain (runs detached inside nvfp4_timeline; every
# step waits for idle GPUs through run_m16_taskshape_h20.sh). R = results root.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
R=${R:-/raid/kimi/results8}
export MARKER=${MARKER:-}
NPASS=${NPASS:-3}           # independent customer-method captures per knob value
SKIP_BASE_P23=${SKIP_BASE_P23:-0}
RUN="bash scripts/run_m16_taskshape_h20.sh"
FUSED_ONLY=(SCOPES="e2e mega" BACKENDS=fused)
MEGA_ONLY=(SCOPES=mega BACKENDS=fused)
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain2.log"; }
log "start"
# Task A: remaining baseline passes (pass 1 is in base_p1/p1)
[ "$SKIP_BASE_P23" = 1 ] || { $RUN capture "$R/base_p23" 2 "2 4 8 16"; log "base_p23 done"; }
# Knob correctness: T=2 8 8 16 both quants (+ mxfp4 128 512)
BIG=1 $RUN corr "$R/corr_l1all.log" DG_FP4_SPLITK_L1_ALL=1; log "corr l1all $(tail -1 "$R/corr_l1all.log")"
BIG=1 $RUN corr "$R/corr_l2all.log" DG_FP4_SPLITK_L2_ALL=1; log "corr l2all $(tail -1 "$R/corr_l2all.log")"
$RUN corr "$R/corr_both.log" DG_FP4_SPLITK_L1_ALL=1 DG_FP4_SPLITK_L2_ALL=1; log "corr both $(tail -1 "$R/corr_both.log")"
# Task B customer-method A/B at M=16 (fused only, 3 independent captures each, same session)
$RUN capture "$R/m16_base" "$NPASS" 16 "${FUSED_ONLY[@]}"; log "m16_base done"
$RUN capture "$R/m16_l1all" "$NPASS" 16 DG_FP4_SPLITK_L1_ALL=1 "${FUSED_ONLY[@]}"; log "m16_l1all done"
$RUN capture "$R/m16_l2all" "$NPASS" 16 DG_FP4_SPLITK_L2_ALL=1 "${FUSED_ONLY[@]}"; log "m16_l2all done"
$RUN capture "$R/m16_both" "$NPASS" 16 DG_FP4_SPLITK_L1_ALL=1 DG_FP4_SPLITK_L2_ALL=1 "${FUSED_ONLY[@]}"; log "m16_both done"
# Direct (skew-free) timing: host barrier before every launch
$RUN capture "$R/hb_base" 1 "2 4 8 16" DG_PROFILE_HOST_BARRIER=1 "${MEGA_ONLY[@]}"; log "hb_base done"
$RUN capture "$R/hb_l1all" 2 16 DG_PROFILE_HOST_BARRIER=1 DG_FP4_SPLITK_L1_ALL=1 "${MEGA_ONLY[@]}"; log "hb_l1all done"
$RUN capture "$R/hb_l2all" 2 16 DG_PROFILE_HOST_BARRIER=1 DG_FP4_SPLITK_L2_ALL=1 "${MEGA_ONLY[@]}"; log "hb_l2all done"
$RUN capture "$R/hb_both" 2 16 DG_PROFILE_HOST_BARRIER=1 DG_FP4_SPLITK_L1_ALL=1 DG_FP4_SPLITK_L2_ALL=1 "${MEGA_ONLY[@]}"; log "hb_both done"
$RUN capture "$R/hb_base16" 2 16 DG_PROFILE_HOST_BARRIER=1 "${MEGA_ONLY[@]}"; log "hb_base16 done"
log "CHAIN2_DONE"
