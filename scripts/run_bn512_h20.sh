#!/usr/bin/env bash
# H20 wide-task (DG_FP4_L1_BN / DG_FP4_L2_BN = 512) validation + measurement runner
# (inside four_api_build, host .7). Every GPU run waits for idle GPUs and for no
# FOREIGN *_RUNNING marker, holds MARKER while running, is wrapped in timeout 900
# (captures: 3600) and never touches the clocks (verify mode).
#   corr    <log> <tokens...> -- [ENV=..]      : mxfp4+qoq fused correctness per T
#   corrbig <log> -- [ENV=..]                  : mxfp4 fused T=128 512
#   stress  <log> <M> -- [ENV=..]              : 200-iter graph-replay probe, mxfp4 + qoq
#   probe   <tag> <M> -- [ENV=..]              : phase-stamp probe mxfp4 + qoq (+ PROBE_TASKLOG=1 pass)
#   capture <outdir> <tokens_list> -- [ENV=..] : one official capture (fused only, e2e + mega)
#   chain   <results_root>                     : the whole A/B (see below)
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0a}
TR=$(command -v torchrun)
MARKER=${MARKER:-/raid/kimi/results/CAPTURE_BN512_RUNNING}
MODE=${1:-}; shift || true

foreign_marker() {
  local f
  for f in "$(dirname "$MARKER")"/*_RUNNING; do
    [ -e "$f" ] || continue
    [ "$f" = "$MARKER" ] || return 0
  done
  return 1
}
hold_marker() { echo "$$ $(date -u +%FT%TZ) $*" > "$MARKER"; }
drop_marker() { rm -f "$MARKER"; }
trap drop_marker EXIT
wait_idle() {
  for _ in $(seq 1 4320); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 64 {n++} END {print n+0}')
    if [ "$busy" = 0 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && ! foreign_marker; then
      hold_marker "$MODE $*"; return 0
    fi
    drop_marker; sleep 10
  done
  echo "GPUs busy for 12 h, giving up" >&2; return 1
}
# split "<positional...> -- ENV=.." into POS / ENVS
POS=(); ENVS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; ENVS=("$@"); break; fi
  POS+=("$1"); shift
done
run_env() { env "${ENVS[@]}" "$@"; }

case "$MODE" in
  corr)
    LOG=${POS[0]}; : > "$LOG"; rc=0
    for T in "${POS[@]:1}"; do
      echo "--- mxfp4+qoq fused T=$T (${ENVS[*]:-})" >> "$LOG"
      wait_idle "corr T=$T"
      run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
        --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1 || rc=1
      grep -E "cos_min|Error|error|Traceback|assert" "$LOG" | tail -3
    done
    echo "CORR_RC=$rc" >> "$LOG"; drop_marker ;;
  corrbig)
    LOG=${POS[0]}; : > "$LOG"; rc=0
    for T in 128 512; do
      echo "--- mxfp4 fused T=$T (${ENVS[*]:-})" >> "$LOG"
      wait_idle "corrbig T=$T"
      run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
        --apis mxfp4_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1 || rc=1
    done
    echo "CORR_RC=$rc" >> "$LOG"; drop_marker ;;
  stress)
    LOG=${POS[0]}; M=${POS[1]}; : > "$LOG"; rc=0
    for Q in mxfp4 qoq; do
      echo "--- stress $Q M=$M iters=200 (${ENVS[*]:-})" >> "$LOG"
      wait_idle "stress $Q"
      run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py \
        --quant "$Q" --global-tokens "$M" --iters 200 > "${LOG%.log}_$Q.log" 2>&1 || rc=1
      grep -E "CUDA-event wall|Error|error|NaN|Traceback|timeout|TRAP" "${LOG%.log}_$Q.log" | head -5 >> "$LOG"
    done
    echo "STRESS_RC=$rc" >> "$LOG"; drop_marker ;;
  probe)
    TAG=${POS[0]}; M=${POS[1]}
    for Q in mxfp4 qoq; do
      wait_idle "probe $Q"
      run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py \
        --quant "$Q" --global-tokens "$M" --iters 20 > "probe_${Q}_m${M}_${TAG}.log" 2>&1
      echo "PROBE_EXIT=$?" >> "probe_${Q}_m${M}_${TAG}.log"
      wait_idle "probe tasklog $Q"
      PROBE_TASKLOG=1 run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py \
        --quant "$Q" --global-tokens "$M" --iters 20 > "probe_${Q}_m${M}_${TAG}_tasklog.log" 2>&1
      echo "PROBE_EXIT=$?" >> "probe_${Q}_m${M}_${TAG}_tasklog.log"
    done
    drop_marker ;;
  capture)
    OUTDIR=${POS[0]}; TOK=${POS[1]}
    mkdir -p "$(dirname "$OUTDIR")"
    wait_idle "capture $OUTDIR"
    run_env TOKENS_LIST="$TOK" OUT="$OUTDIR" LOCK_SM_CLOCK_MHZ=1830 FORCE=1 SCOPES="e2e mega" BACKENDS=fused \
      timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUTDIR.log" 2>&1
    echo "CAPTURE_RC=$?" >> "$OUTDIR.log"
    drop_marker ;;
  chain)
    R=${POS[0]}; mkdir -p "$R"; C="$R/chain.log"
    log() { echo "$(date -u +%FT%TZ) $*" >> "$C"; }
    ON=(DG_FP4_L1_BN=512 DG_FP4_L2_BN=512)
    L1ONLY=(DG_FP4_L1_BN=512 DG_FP4_L2_BN=256)
    L2ONLY=(DG_FP4_L1_BN=256 DG_FP4_L2_BN=512)
    NPASS=${NPASS:-5}
    log "start $(git rev-parse --short HEAD)"
    # A. bring-up correctness (spin timeout traps instead of hanging), gate proof, variants
    # (--tokens is per rank: 2 -> 16 global tokens == the wide tier; 8 8 16 -> M=64/128, gate off)
    bash "$0" corr "$R/corr_on_m16.log" 2 -- "${ON[@]}" DG_FP4_SPIN_TIMEOUT=1; log "corr on M16: $(tail -1 "$R/corr_on_m16.log")"
    grep -q "CORR_RC=0" "$R/corr_on_m16.log" || { log "ABORT: bring-up failed"; exit 1; }
    bash "$0" corr "$R/corr_on_t8_8_16.log" 8 8 16 -- "${ON[@]}"; log "corr on t8 8 16 (gate): $(tail -1 "$R/corr_on_t8_8_16.log")"
    bash "$0" corr "$R/corr_l1only_m16.log" 2 -- "${L1ONLY[@]}" DG_FP4_SPIN_TIMEOUT=1; log "corr l1only: $(tail -1 "$R/corr_l1only_m16.log")"
    bash "$0" corr "$R/corr_l2only_m16.log" 2 -- "${L2ONLY[@]}" DG_FP4_SPIN_TIMEOUT=1; log "corr l2only: $(tail -1 "$R/corr_l2only_m16.log")"
    bash "$0" corrbig "$R/corr_on_big.log" -- "${ON[@]}"; log "corr on 128/512: $(tail -1 "$R/corr_on_big.log")"
    # B. stress
    bash "$0" stress "$R/stress_on_m16.log" 16 -- "${ON[@]}" DG_FP4_SPIN_TIMEOUT=1; log "stress on: $(tail -1 "$R/stress_on_m16.log")"
    # C. probes (off / on / l1only), incl. per-task log passes
    bash "$0" probe bn_off 16 -- DG_FP4_L1_BN=256 DG_FP4_L2_BN=256; log "probe off done"
    bash "$0" probe bn_on 16 -- "${ON[@]}"; log "probe on done"
    bash "$0" probe bn_l1only 16 -- "${L1ONLY[@]}"; log "probe l1only done"
    mkdir -p "$R/probes"; mv probe_*_m16_bn_*.log "$R/probes/" 2>/dev/null
    # D. customer-method captures, interleaved passes off / on / l1only
    for p in $(seq 1 "$NPASS"); do
      bash "$0" capture "$R/cap_off/p$p" 16 -- DG_FP4_L1_BN=256 DG_FP4_L2_BN=256; log "cap off p$p: $(tail -1 "$R/cap_off/p$p.log")"
      bash "$0" capture "$R/cap_on/p$p" 16 -- "${ON[@]}"; log "cap on p$p: $(tail -1 "$R/cap_on/p$p.log")"
      bash "$0" capture "$R/cap_l1only/p$p" 16 -- "${L1ONLY[@]}"; log "cap l1only p$p: $(tail -1 "$R/cap_l1only/p$p.log")"
    done
    for k in off on l1only; do
      python3 scripts/summarize_m16_passes.py "$R/cap_$k"/p* > "$R/cap_$k/SUMMARY.md" 2>> "$C"
    done
    log "CHAIN_DONE" ;;
  *) echo "usage: see header" >&2; exit 2 ;;
esac
