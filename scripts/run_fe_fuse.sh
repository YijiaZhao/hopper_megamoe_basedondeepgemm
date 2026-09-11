#!/usr/bin/env bash
# Fused Fable frontend (DG_FP4_FUSE_FE, docs/fe_into_mega_design.md) validation + A/B runner
# (inside four_api_build on .7). Every GPU run waits for idle GPUs (no compute process) and
# for no FOREIGN *_RUNNING marker, holds MARKER while running, is wrapped in timeout 900
# (captures 3600) and leaves the clocks alone (capture verifies 1830 MHz).
#   corr    <log> <tokens...> -- [ENV=..] : plain mxfp4+qoq fused correctness per T (knob inert w/o frontend)
#   corrfe  <log> <tokens...> -- [ENV=..] : --frontend fused: FE+Mega vs fused-FE bit-equality + reference (T <= 2)
#   corrref <log> <tokens...> -- [ENV=..] : --frontend fe: standalone FE routing + Mega vs the exact reference (per-slot check)
#   stress  <log> <M> -- [ENV=..]         : 200-iter graph replay of FE / Mega / FE+Mega / FEinMega, both quants
#   bench   <log> <M...> -- [ENV=..]      : CUDA-event E2E, FE+Mega (knob 0) vs FEinMega (knob 1), n=100, both quants
#   probe   <tag> <M> -- [ENV=..]         : phase-stamp probe with the FE fused in (both quants) -> $RES/fefuse/probe_<tag>_*.log
#   debug   <log> <M> -- [ENV=..]         : tests/debug_fe_fuse.py (determinism / equality diagnostics), both quants
#   capture <outdir> <tokens_list> -- [ENV=..] : one official e2e/fused capture (customer method)
#   chain   <results_root>                : the whole A/B
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-/raid/kimi/dg_dev/third-party/cutlass/include}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0a}
export DG_FE_TINYM=1
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
TR=$(command -v torchrun)
RES=${RES:-/raid/kimi/results}
MARKER=${MARKER:-$RES/CAPTURE_FEFUSE_RUNNING}
MODE=${1:-}; shift || true

# A foreign marker older than STALE_MARKER_MIN minutes (default 240) with no GPU compute
# process is treated as left behind; it is never deleted. (30 min was too short: another
# agent held its marker across a long chain with idle gaps and we collided with it.)
STALE_MARKER_MIN=${STALE_MARKER_MIN:-240}
foreign_marker() {
  local f age
  for f in "$(dirname "$MARKER")"/*_RUNNING; do
    [ -e "$f" ] || continue
    [ "$f" = "$MARKER" ] && continue
    age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
    if [ "$age" -ge "$STALE_MARKER_MIN" ]; then
      echo "$(date -u +%FT%TZ) ignoring stale foreign marker $f (${age} min, no GPU process)" >&2
      continue
    fi
    return 0
  done
  return 1
}
hold_marker() { echo "$$ $(date -u +%FT%TZ) $*" > "$MARKER"; }
drop_marker() { rm -f "$MARKER"; }
trap drop_marker EXIT
wait_idle() {
  local quiet=0
  for _ in $(seq 1 4320); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && ! foreign_marker; then
      quiet=$((quiet + 1))
      if [ "$quiet" -ge 3 ]; then hold_marker "$MODE $*"; return 0; fi
    else
      quiet=0
    fi
    drop_marker; sleep 10
  done
  echo "GPUs busy for 12 h, giving up" >&2; return 1
}
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
      echo "--- mxfp4+qoq fused T=$T DG_FP4_FUSE_FE=1 (${ENVS[*]:-})" >> "$LOG"
      wait_idle "corr T=$T"
      DG_FP4_FUSE_FE=1 run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
        --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" >> "$LOG" 2>&1 || rc=1
      grep -E "RESULT|Error|error|Traceback|assert" "$LOG" | tail -3
    done
    echo "CORR_RC=$rc" >> "$LOG"; drop_marker ;;
  corrfe)
    LOG=${POS[0]}; : > "$LOG"; rc=0
    for T in "${POS[@]:1}"; do
      echo "--- mxfp4+qoq fused --frontend fused T=$T (${ENVS[*]:-})" >> "$LOG"
      wait_idle "corrfe T=$T"
      DG_FP4_FUSE_FE=1 run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
        --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" --frontend fused >> "$LOG" 2>&1 || rc=1
      grep -E "RESULT|FUSED_FE_EQUALITY|Error|error|Traceback|assert" "$LOG" | tail -4
    done
    echo "CORRFE_RC=$rc" >> "$LOG"; drop_marker ;;
  corrref)
    LOG=${POS[0]}; : > "$LOG"; rc=0
    for T in "${POS[@]:1}"; do
      echo "--- mxfp4+qoq fused --frontend fe T=$T (${ENVS[*]:-})" >> "$LOG"
      wait_idle "corrref T=$T"
      run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
        --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" --frontend fe >> "$LOG" 2>&1 || rc=1
    done
    echo "CORRREF_RC=$rc" >> "$LOG"; drop_marker ;;
  stress)
    LOG=${POS[0]}; M=${POS[1]}; : > "$LOG"; rc=0
    for Q in mxfp4 qoq; do
      echo "--- stress $Q M=$M iters=200 DG_FP4_FUSE_FE=1 (${ENVS[*]:-})" >> "$LOG"
      wait_idle "stress $Q"
      DG_FP4_FUSE_FE=1 run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/bench_frontend_tinym.py \
        --quant "$Q" --global-tokens "$M" --iters 200 > "${LOG%.log}_$Q.log" 2>&1 || rc=1
      grep -E "median|Error|error|NaN|Traceback|timeout|TRAP|EXIT" "${LOG%.log}_$Q.log" | head -8 >> "$LOG"
    done
    echo "STRESS_RC=$rc" >> "$LOG"; drop_marker ;;
  bench)
    LOG=${POS[0]}; : > "$LOG"; rc=0
    for M in "${POS[@]:1}"; do
      for Q in mxfp4 qoq; do
        echo "--- bench $Q M=$M DG_FP4_FUSE_FE=1 (${ENVS[*]:-})" >> "$LOG"
        wait_idle "bench $Q M=$M"
        DG_FP4_FUSE_FE=1 run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/bench_frontend_tinym.py \
          --quant "$Q" --global-tokens "$M" --iters 100 >> "$LOG" 2>&1 || rc=1
      done
    done
    echo "BENCH_RC=$rc" >> "$LOG"; drop_marker ;;
  probe)
    TAG=${POS[0]}; M=${POS[1]:-8}
    for Q in mxfp4 qoq; do
      wait_idle "probe $Q M=$M"
      DG_FP4_FUSE_FE=1 run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py \
        --quant "$Q" --global-tokens "$M" --iters 20 --fuse-fe > "$RES/fefuse/probe_${TAG}_${Q}_m$M.log" 2>&1
      echo "PROBE_EXIT=$?" >> "$RES/fefuse/probe_${TAG}_${Q}_m$M.log"
    done
    drop_marker ;;
  debug)
    LOG=${POS[0]}; M=${POS[1]:-8}; : > "$LOG"
    for Q in mxfp4 qoq; do
      echo "--- debug $Q M=$M DG_FP4_FUSE_FE=1 (${ENVS[*]:-})" >> "$LOG"
      wait_idle "debug $Q"
      DG_FP4_FUSE_FE=1 run_env timeout 900 "$TR" --standalone --nproc_per_node=8 tests/debug_fe_fuse.py \
        --quant "$Q" --global-tokens "$M" >> "$LOG" 2>&1
      echo "DEBUG_EXIT=$?" >> "$LOG"
    done
    drop_marker ;;
  capture)
    OUTDIR=${POS[0]}; TOK=${POS[1]}
    mkdir -p "$(dirname "$OUTDIR")"
    wait_idle "capture $OUTDIR"
    run_env TOKENS_LIST="$TOK" OUT="$OUTDIR" LOCK_SM_CLOCK_MHZ=1830 FORCE=1 SCOPES="e2e" BACKENDS=fused \
      timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUTDIR.log" 2>&1
    echo "CAPTURE_RC=$?" >> "$OUTDIR.log"
    drop_marker ;;
  chain)
    R=${POS[0]}; mkdir -p "$R"; C="$R/chain.log"
    log() { echo "$(date -u +%FT%TZ) $*" >> "$C"; }
    NCAP=${NCAP:-3}
    log "start $(git rev-parse --short HEAD)"
    # A. bring-up: fused-FE equality (spin timeouts trap instead of hanging), M=8 and M=16 global
    bash "$0" corrfe "$R/corrfe_t1.log" 1 -- DG_FP4_SPIN_TIMEOUT=1; log "corrfe T=1: $(tail -1 "$R/corrfe_t1.log")"
    grep -q "CORRFE_RC=0" "$R/corrfe_t1.log" || { log "ABORT: bring-up failed"; exit 1; }
    bash "$0" corrfe "$R/corrfe_t2.log" 2 -- DG_FP4_SPIN_TIMEOUT=1; log "corrfe T=2: $(tail -1 "$R/corrfe_t2.log")"
    # B. the knob is inert without frontend tensors: the standard matrix, knob env on
    bash "$0" corr "$R/corr_knob_on.log" 2 8 8 16; log "corr knob on T=2 8 8 16: $(tail -1 "$R/corr_knob_on.log")"
    # C. stress (200-iter graph replay incl. the fused-FE graph)
    for M in 2 8 16; do bash "$0" stress "$R/stress_m$M.log" "$M" -- DG_FP4_SPIN_TIMEOUT=1; log "stress M=$M: $(tail -1 "$R/stress_m$M.log")"; done
    # D. CUDA-event E2E, two passes
    for p in 1 2; do bash "$0" bench "$R/bench_p$p.log" 2 8 16; log "bench p$p: $(tail -1 "$R/bench_p$p.log")"; done
    # E. customer-method captures, interleaved knob 0 / 1
    for p in $(seq 1 "$NCAP"); do
      bash "$0" capture "$R/cap_k0/p$p" "2 8 16" -- DG_FP4_FUSE_FE=0; log "cap k0 p$p: $(tail -1 "$R/cap_k0/p$p.log")"
      bash "$0" capture "$R/cap_k1/p$p" "2 8 16" -- DG_FP4_FUSE_FE=1; log "cap k1 p$p: $(tail -1 "$R/cap_k1/p$p.log")"
    done
    python3 scripts/summarize_knob_captures.py --knob 0 "$R"/cap_k0/p* --knob 1 "$R"/cap_k1/p* --fused-only > "$R/CAPTURE_SUMMARY.md" 2>> "$C"
    log "CHAIN_DONE" ;;
  *) echo "usage: see header" >&2; exit 2 ;;
esac
