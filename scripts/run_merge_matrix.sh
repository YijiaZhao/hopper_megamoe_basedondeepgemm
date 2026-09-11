#!/usr/bin/env bash
# Test matrix for the perf/fe-into-mega merge (H20 .7, container four_api_build, worktree
# /raid/kimi/dg_merge). Modes:
#   chain1            corr (T=2 8 8 16 default knobs) + corrref T=1,2 + revert-proof + corrfe T=2
#                     + mxfp4 T=128/512 + FE equality (swapab default, cc) 500 seeds
#   corr T...         tests/test_four_api_correctness.py both fused apis, default knobs
#   corrref T...      --frontend fe (real routing; exposes the stream-K-slot bug fixed in 6d594ee)
#   revert T          corrref with 6d594ee reverted in the worktree (must FAIL), file restored after
#   corrfe T...       DG_FP4_FUSE_FE=1 --frontend fused (equality)
#   mx T...           mxfp4 only
#   fe78 MMA SEEDS [GPU] [ROWS...]  tests/test_frontend_fe78.py equality (single GPU)
#   capture DIR       customer-method capture SCOPES=e2e BACKENDS=fused TOKENS_LIST="2 8 16"
#   land_diag         triage of the chain4 corrref T=32 qoq failure: default-routing qoq T=64/128,
#                     --frontend fe T=32 under DG_FE_TINYM_MMA=cc, --frontend fe T=8/16
# Every 8-GPU run waits for idle GPUs + no foreign *_RUNNING marker, holds $MARKER (CAPTURE_MERGE_RUNNING).
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
RES=${RES:-/raid/kimi/results/merge}
mkdir -p "$RES"
# Own JIT cache: the JIT key hashes the generated code string, not the included headers, so a
# cache shared with other worktrees could hand back binaries built from other header revisions.
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-$RES/jit}
TR=$(command -v torchrun)
MARKER=${MARKER:-/raid/kimi/results/CAPTURE_MERGE_RUNNING}   # override: MARKER=/raid/kimi/results/CAPTURE_LAND_RUNNING
MODE=${1:-}; shift || true
foreign_marker() {
  local f age
  for f in /raid/kimi/results/*_RUNNING; do
    [ -e "$f" ] || continue
    [ "$f" = "$MARKER" ] && continue
    age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
    [ "$age" -ge 240 ] && continue
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
CORR="$TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py"
summ() { grep -E "cos_min=|SLOT_CHECK|FUSED_FE_EQUALITY|PASS|FAIL|Error|Traceback|AssertionError" "$1" | grep -v '^\s*$' | tail -${2:-6}; }
run_corr() {   # tag, extra args..., env via caller
  local tag=$1; shift
  local log="$RES/$tag.log"
  wait_idle "$tag"
  timeout 900 $CORR "$@" > "$log" 2>&1; local rc=$?
  drop_marker
  echo "== $tag rc=$rc"; summ "$log"
  return $rc
}
case "$MODE" in
  corr)    rc=0; for T in "$@"; do run_corr "corr_t$T" --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" || rc=1; done; echo "CORR_RC=$rc" ;;
  corrref) rc=0; for T in "$@"; do run_corr "corrref_t$T" --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" --frontend fe --cosine-min "${COSMIN:-0.9999}" || rc=1; done; echo "CORRREF_RC=$rc" ;;
  corrfe)  rc=0; for T in "$@"; do DG_FP4_FUSE_FE=1 run_corr "corrfe_t$T" --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" --frontend fused || rc=1; done; echo "CORRFE_RC=$rc" ;;
  mx)      rc=0; for T in "$@"; do run_corr "mx_t$T" --apis mxfp4_mega_moe_fused --tokens "$T" || rc=1; done; echo "MX_RC=$rc" ;;
  revert)
    T=${1:-1}
    F=deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused_body.inl
    git diff --quiet -- "$F" || { echo "worktree dirty: $F"; exit 2; }
    git show 6d594ee -- "$F" | git apply -R || { echo "reverse-apply of 6d594ee failed"; exit 2; }
    echo "6d594ee reverted in $F ($(git diff --stat -- "$F" | tail -1))"
    DG_JIT_CACHE_DIR="$RES/jit_revert" DG_FP4_SPIN_TIMEOUT=1 run_corr "revert_corrref_t$T" --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens "$T" --frontend fe --cosine-min 0.9999
    rc=$?
    git checkout -- "$F"; echo "restored: $(git status --short -- "$F" | wc -l) dirty files (expect 0)"
    if [ "$rc" -ne 0 ]; then echo "REVERT_PROOF=FAILS_AS_EXPECTED"; else echo "REVERT_PROOF=UNEXPECTED_PASS"; fi ;;
  fe78)
    MM=$1; SEEDS=$2; GPU=${3:-7}; OFF=${4:-0}; shift 4 || shift $#
    ROWS=${*:-1 2 8 16}
    case "$MM" in cc) EXTRA="--grid auto --mma cc" ;; auto) EXTRA="--mma auto" ;; *) EXTRA="--grid 96 --mma swapab --wlayout fragment" ;; esac
    log="$RES/fe78_${MM}_s${SEEDS}_o$OFF.log"
    CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 tests/test_frontend_fe78.py --seeds "$SEEDS" --seed-offset "$OFF" $EXTRA --rows $ROWS > "$log" 2>&1
    echo "== fe78 $MM seeds=$SEEDS offset=$OFF rows=$ROWS rc=$?"; grep -E "new-scheme|full-K vs legacy|MISMATCH|WEIGHT DIFF|^PASS|^FAIL|Error" "$log" | tail -5 ;;
  capture)
    OUTDIR=$1; TOK=${2:-"2 8 16"}
    mkdir -p "$(dirname "$OUTDIR")"
    wait_idle "capture $OUTDIR"
    TOKENS_LIST="$TOK" OUT="$OUTDIR" LOCK_SM_CLOCK_MHZ=1830 FORCE=1 SCOPES="e2e" BACKENDS=fused GPU_IDLE_LIMIT_MIB=${GPU_IDLE_LIMIT_MIB:-256} \
      timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUTDIR.log" 2>&1
    echo "CAPTURE_RC=$?" >> "$OUTDIR.log"; tail -2 "$OUTDIR.log"; drop_marker ;;
  chain1)
    C="$RES/chain1.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      bash "$0" corr 2 8 8 16
      bash "$0" corrref 1 2
      bash "$0" revert 1
      bash "$0" corrfe 2
      bash "$0" mx 128 512
      wait_idle fe78; drop_marker
      bash "$0" fe78 swapab 500 6 0 > "$RES/fe78_swapab.out" 2>&1 &
      bash "$0" fe78 cc 500 7 0 > "$RES/fe78_cc.out" 2>&1 &
      wait; cat "$RES/fe78_swapab.out" "$RES/fe78_cc.out"
      echo "CHAIN1_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  chain2)   # after the FE default flip (DG_FE_TINYM_MMA=auto): FE equality of the defaults, 8-rank
            # correctness (T=1 2 -> cc router; 8 16 -> swapab), --frontend fe at T=1/2, 3 customer captures
    C="$RES/chain2.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      wait_idle fe78auto; drop_marker
      bash "$0" fe78 auto 500 6 0 1 2 8 16 > "$RES/fe78_auto_a.out" 2>&1 &
      bash "$0" fe78 auto 500 7 500 1 2 8 16 > "$RES/fe78_auto_b.out" 2>&1 &
      wait; cat "$RES/fe78_auto_a.out" "$RES/fe78_auto_b.out"
      bash "$0" corr 1 2 8 16
      bash "$0" corrref 1 2
      for p in 1 2 3; do bash "$0" capture "$RES/cap/p$p" "2 8 16"; done
      echo "CHAIN2_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  chain3)   # final defaults (cc + auto grid + L2 persist for rows <= 2): 8-rank correctness matrix + 3 customer captures
    C="$RES/chain3.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      bash "$0" corr 1 2 8 16
      bash "$0" corrref 1 2
      bash "$0" corrfe 2
      bash "$0" mx 128 512
      for p in 1 2 3; do bash "$0" capture "$RES/cap_final/p$p" "2 8 16"; done
      echo "CHAIN3_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  chain4)   # default = swapab + fragment (96 grid) for every row count: 8-rank matrix (+ --frontend fe at 32 rows = non-tiny FE) + 3 customer captures
    C="$RES/chain4.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      bash "$0" corr 1 2 8 16
      bash "$0" corrref 1 2
      COSMIN=0.999 bash "$0" corrref 32
      bash "$0" corrfe 2
      bash "$0" mx 128 512
      for p in 1 2 3; do bash "$0" capture "$RES/cap_swapab/p$p" "2 8 16"; done
      echo "CHAIN4_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  corrq)   rc=0; for T in "$@"; do run_corr "corrq_t$T" --apis qoq_mega_moe_fused --tokens "$T" || rc=1; done; echo "CORRQ_RC=$rc" ;;
  chain5)   # round-3 cc vs swapab under the customer method (M=8 = 1 row/rank) + qoq 32-row diagnostics
    C="$RES/chain5.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      DG_FE_TINYM_MMA=cc bash "$0" capture "$RES/cap_ab3/cc_r3" 8
      DG_FE_TINYM_MMA=swapab bash "$0" capture "$RES/cap_ab3/swapab" 8
      DG_FE_TINYM_MMA=cc bash "$0" capture "$RES/cap_ab3/cc_r3_b" 8
      DG_FE_TINYM_MMA=swapab bash "$0" capture "$RES/cap_ab3/swapab_b" 8
      for d in cc_r3 swapab cc_r3_b swapab_b; do echo "## $d"; cut -d, -f1-4,6-8 "$RES/cap_ab3/$d/TIMELINE_LAST3.csv"; done
      bash "$0" corrq 32
      COSMIN=0.999 bash "$0" corrref 32
      echo "CHAIN5_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  chain6)   # final defaults (round-3 cc for rows <= 2, swapab otherwise): 8-rank gates + 3 customer captures
    C="$RES/chain6.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      bash "$0" corr 1 2 8 16
      bash "$0" corrref 1 2
      bash "$0" corrfe 2
      bash "$0" mx 128 512
      for p in 1 2 3; do bash "$0" capture "$RES/cap_cc3/p$p" "2 8 16"; done
      for p in 1 2 3; do echo "## p$p"; cut -d, -f1-4,6-8 "$RES/cap_cc3/p$p/TIMELINE_LAST3.csv"; done
      echo "CHAIN6_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  land_diag)  # chain4 corrref T=32 qoq failure triage (see README FE section)
    C="$RES/land_diag.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      bash "$0" corrq 64 128
      DG_FE_TINYM_MMA=cc COSMIN=0.999 bash "$0" corrref 32
      COSMIN=0.999 bash "$0" corrref 8 16
      for kv in DG_FP4_QOQ_INLINE_S2=0 DG_FP4_SPLITK_L1=0 DG_FP4_LEAN_ROUTING=0 DG_FP4_FINE_COMBINE=0 DG_FP4_COMBINE_DYNAMIC=0; do bash "$0" qknob "$kv" 32; done
      echo "LAND_DIAG_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  qknob)   # qoq --frontend fe under one env override: qknob NAME=VAL [T=32] (tier isolation of the 32-row qoq failure)
    KV=$1; T=${2:-32}; export "$KV"
    run_corr "qknob_${KV//=/_}_t$T" --apis qoq_mega_moe_fused --tokens "$T" --frontend fe --cosine-min 0.999; echo "QKNOB_RC=$? $KV" ;;
  land_ab)    # round-3 cc (3.84 us stamps) vs swapab+fragment under the customer method, M=8 = 1 row/rank, 2 x 2 captures
    C="$RES/land_ab.log"; : > "$C"
    { echo "start $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
      DG_FE_TINYM_MMA=cc bash "$0" capture "$RES/cap_land_ab/cc_r3" 8
      DG_FE_TINYM_MMA=swapab bash "$0" capture "$RES/cap_land_ab/swapab" 8
      DG_FE_TINYM_MMA=cc bash "$0" capture "$RES/cap_land_ab/cc_r3_b" 8
      DG_FE_TINYM_MMA=swapab bash "$0" capture "$RES/cap_land_ab/swapab_b" 8
      for d in cc_r3 swapab cc_r3_b swapab_b; do echo "## $d"; cut -d, -f1-4,6-8 "$RES/cap_land_ab/$d/TIMELINE_LAST3.csv"; done
      echo "LAND_AB_DONE $(date -u +%FT%TZ)"; } >> "$C" 2>&1 ;;
  stop)     # stop OUR OWN jobs only: processes whose cwd is this worktree (run inside the same container)
    for pid in $(pgrep -f "run_merge_matrix.sh|capture_four_api_h20_timelines.sh|nsys profile|profile_four_api_h20.py|torchrun|test_four_api_correctness.py|test_frontend_fe78.py"); do
      [ "$pid" = "$$" ] && continue
      [ "$(readlink /proc/$pid/cwd 2>/dev/null)" = "$ROOT" ] || continue
      echo "kill $pid $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-120)"; kill "$pid" 2>/dev/null
    done
    sleep 8; drop_marker; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader ;;
  *) echo "usage: see header" >&2; exit 2 ;;
esac
