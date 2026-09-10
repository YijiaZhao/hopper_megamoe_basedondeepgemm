#!/usr/bin/env bash
# Third math WG (DG_FP4_MATH_WGS=3, docs/third_math_wg.md phase 2) validation + perf,
# inside four_api_build on .7:
#   MODE=corr|stress|sass|probe|event|nsys|all [TAG=..] bash scripts/wg3_validate.sh
# corr   : tests/test_four_api_correctness.py, WGS=3 (spin timeout on), mxfp4 + qoq at
#          T = 2 8 8 16 tokens/rank, plus --global-tokens 2 (stream-K short segments) and
#          the mxfp4 128 / 512-token tiers (host falls back to 2 WGs). ptxas -v captured.
# stress : 200 graph replays at M=8 (bench_frontend_tinym.py --iters 200), WGS=3.
# sass   : GMMA / DEPBAR / spill counts of the JIT cubins (serialisation check).
# probe  : scripts/run_probe.sh mxfp4|qoq 8 16, WGS 2 vs 3, two passes (pass 2 reversed).
# event  : bench_frontend_tinym.py --iters 100 Mega / FE+Mega medians, WGS 2 vs 3, 2 passes.
# nsys   : one customer-method capture per knob (TOKENS_LIST="2 4 8 16", e2e+mega fused).
# GPU discipline: waits for idle GPUs and no foreign /raid/kimi/results/*_RUNNING marker,
# holds /raid/kimi/results/CAPTURE_WG3_RUNNING while running, never kills anything.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-all}
TAG=${TAG:-}
# MODE=a,b,c runs the modes one after another (each holds the marker for its duration).
if [[ "$MODE" == *,* ]]; then
  for m in ${MODE//,/ }; do MODE=$m bash "$0" || true; done
  exit 0
fi
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-/raid/kimi/dg_dev/third-party/cutlass/include}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
TR=/usr/local/bin/torchrun
RES=/raid/kimi/results
MARK=$RES/CAPTURE_WG3_RUNNING
OUTD=${OUTD:-/raid/kimi/wg3_val$TAG}
mkdir -p "$OUTD"
LOG=$OUTD/wg3_validate.log
CACHE=${DG_JIT_CACHE_DIR:-$OUTD/.jit}
export DG_JIT_CACHE_DIR="$CACHE"

wait_idle() {
  local i
  for i in $(seq 1 720); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] &&
       [ -z "$(ls "$RES"/*_RUNNING 2>/dev/null | grep -v CAPTURE_WG3_RUNNING)" ]; then
      return 0
    fi
    sleep 5
  done
  echo "GPUs busy / foreign marker present after 1 h" | tee -a "$LOG" >&2
  return 1
}
hold() { echo "$$ $(date -u +%FT%TZ) wg3 $MODE" > "$MARK"; }
# Another wg3_validate instance (e.g. an earlier MODE) may still hold the marker.
if [ "${WAIT_FOR_MARKER:-1}" = 1 ]; then
  while [ -e "$MARK" ]; do sleep 10; done
fi
trap 'rm -f "$MARK"' EXIT
hold

run_corr() {  # $1 api, $2 tokens, $3 cos min, $4 label, rest: env
  local api=$1 T=$2 cmin=$3 label=$4; shift 4
  local extra=""
  case "$T" in g2) extra="--tokens 1 --global-tokens 2" ;; *) extra="--tokens $T" ;; esac
  wait_idle || return 1
  echo "--- CORR $label $api T=$T cos>=$cmin ($*)" >> "$LOG"
  env "$@" timeout 600 $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis "$api" $extra --cosine-min "$cmin" > "$OUTD/corr_${label}.log" 2>&1
  local rc=$?
  grep -h "ptxas info\|bytes spill\|bytes stack\|Used [0-9]* registers\|C75[0-9]*\|RESULT\|Error\|error\|TIMEOUT\|timeout" "$OUTD/corr_${label}.log" |
    grep -v "^$" | sed -E 's/for function .*$//' | sort -u | cut -c1-200 >> "$LOG"
  echo "EXIT=$rc" >> "$LOG"
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  echo "=== CORR build $(git rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$LOG"
  W3="DG_FP4_MATH_WGS=3 DG_FP4_SPIN_TIMEOUT=1 DG_JIT_PTXAS_VERBOSE=1"
  i=0
  for T in 2 8 8 16 g2; do
    i=$((i + 1))
    run_corr mxfp4_mega_moe_fused $T 0.99998 "mxfp4_w3_T${T}_$i" $W3
    run_corr qoq_mega_moe_fused $T 0.99993 "qoq_w3_T${T}_$i" $W3
  done
  # large-M tiers: host must keep two WGs (kernel name without _wg3)
  run_corr mxfp4_mega_moe_fused 16 0.99 "mxfp4_w3env_T16_largeM" $W3
  run_corr mxfp4_mega_moe_fused 64 0.99 "mxfp4_w3env_T64_largeM" $W3
  # WGS=2 reference numerics (same seed) at the same points, plus the two-WG QoQ
  # per-block promote (DG_FP4_QOQ_INLINE_S2=0: the arithmetic the 3-WG QoQ loop uses)
  for T in 2 8 16 g2; do
    run_corr mxfp4_mega_moe_fused $T 0.99 "mxfp4_w2_T$T" DG_FP4_MATH_WGS=2 DG_JIT_PTXAS_VERBOSE=1
    run_corr qoq_mega_moe_fused $T 0.99 "qoq_w2_T$T" DG_FP4_MATH_WGS=2 DG_JIT_PTXAS_VERBOSE=1
    run_corr qoq_mega_moe_fused $T 0.99 "qoq_w2_noqis2_T$T" DG_FP4_MATH_WGS=2 DG_FP4_QOQ_INLINE_S2=0 DG_JIT_PTXAS_VERBOSE=1
  done
  echo "kernels built: $(ls "$CACHE" 2>/dev/null | grep -c kernel)" >> "$LOG"
  ls "$CACHE" 2>/dev/null | grep kernel | sed 's/\.[0-9a-f]*$//' | sort -u >> "$LOG"
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = corrq ]; then
  # quick numerics + ptxas check without the spin-timeout trap (which adds a CALL: C7510)
  echo "=== CORRQ build $(git rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$LOG"
  run_corr mxfp4_mega_moe_fused 8 0.99998 "q_mxfp4_w3_T8" DG_FP4_MATH_WGS=3 DG_JIT_PTXAS_VERBOSE=1
  run_corr qoq_mega_moe_fused 8 0.99993 "q_qoq_w3_T8" DG_FP4_MATH_WGS=3 DG_JIT_PTXAS_VERBOSE=1
  run_corr mxfp4_mega_moe_fused g2 0.99998 "q_mxfp4_w3_g2" DG_FP4_MATH_WGS=3 DG_JIT_PTXAS_VERBOSE=1
  run_corr mxfp4_mega_moe_fused g2 0.99 "q_mxfp4_w2_g2" DG_FP4_MATH_WGS=2 DG_JIT_PTXAS_VERBOSE=1
  run_corr qoq_mega_moe_fused 16 0.99 "q_qoq_w2_T16" DG_FP4_MATH_WGS=2 DG_JIT_PTXAS_VERBOSE=1
  run_corr qoq_mega_moe_fused 16 0.99 "q_qoq_w2_noqis2_T16" DG_FP4_MATH_WGS=2 DG_FP4_QOQ_INLINE_S2=0 DG_JIT_PTXAS_VERBOSE=1
  echo ALL_CORRQ_DONE >> "$LOG"
fi

if [ "$MODE" = stress ] || [ "$MODE" = all ]; then
  for q in mxfp4 qoq; do
    wait_idle || exit 1
    echo "--- STRESS $q M=8 200 graph replays WGS=3" >> "$LOG"
    env DG_FP4_MATH_WGS=3 DG_FP4_SPIN_TIMEOUT=1 timeout 600 $TR --standalone --nproc_per_node=8 \
      tests/bench_frontend_tinym.py --quant $q --global-tokens 8 --iters 200 > "$OUTD/stress_$q.log" 2>&1
    echo "EXIT=$?" >> "$LOG"
    grep -E "median|Error|error|TIMEOUT|timeout" "$OUTD/stress_$q.log" | head -8 >> "$LOG"
  done
  echo ALL_STRESS_DONE >> "$LOG"
fi

if [ "$MODE" = sass ] || [ "$MODE" = all ]; then
  echo "--- SASS (cubins under $CACHE)" >> "$LOG"
  for cb in $(find "$CACHE" -name '*.cubin' | sort); do
    echo "cubin: $cb" >> "$LOG"
    cuobjdump -sass "$cb" > "$OUTD/sass_tmp.txt" 2>/dev/null
    echo "QGMMA=$(grep -c QGMMA "$OUTD/sass_tmp.txt") IGMMA=$(grep -c IGMMA "$OUTD/sass_tmp.txt") DEPBAR0=$(grep -c 'DEPBAR.LE gsb0, 0x0' "$OUTD/sass_tmp.txt") DEPBAR1=$(grep -c 'DEPBAR.LE gsb0, 0x1' "$OUTD/sass_tmp.txt") ARRIVE=$(grep -c 'WARPGROUP.ARRIVE' "$OUTD/sass_tmp.txt") LDL=$(grep -c ' LDL' "$OUTD/sass_tmp.txt") STL=$(grep -c ' STL' "$OUTD/sass_tmp.txt") SETMAXREG=$(grep -o 'SETMAXREG[^;]*' "$OUTD/sass_tmp.txt" | tr -s ' ' | tr '\n' ',')" >> "$LOG"
  done
  echo ALL_SASS_DONE >> "$LOG"
fi

probe_one() {  # $1 quant, $2 M, $3 wgs, $4 pass
  local q=$1 M=$2 w=$3 p=$4
  wait_idle || return 1
  env DG_FP4_MATH_WGS=$w LOG_TAG="_wg${w}_p${p}$TAG" timeout 600 bash scripts/run_probe.sh "$q" "$M" > /dev/null 2>&1
  echo "--- PROBE $q M=$M WGS=$w pass=$p" >> "$LOG"
  grep -E "^ *(1|2|3|4|5|7|16|17|18|19|21|22|30|31) |PROBE_EXIT|NO-STAMPS" "probe_${q}_m${M}_wg${w}_p${p}$TAG.log" | cut -c1-80 >> "$LOG"
  mv -f "probe_${q}_m${M}_wg${w}_p${p}$TAG.log" "$OUTD/" 2>/dev/null
}
if [ "$MODE" = probe ] || [ "$MODE" = all ]; then
  for q in mxfp4 qoq; do for M in 8 16; do for w in 2 3; do probe_one $q $M $w 1; done; done; done
  for q in qoq mxfp4; do for M in 16 8; do for w in 3 2; do probe_one $q $M $w 2; done; done; done
  echo ALL_PROBE_DONE >> "$LOG"
fi

event_one() {  # $1 quant, $2 M, $3 wgs, $4 pass
  local q=$1 M=$2 w=$3 p=$4
  wait_idle || return 1
  echo "--- EVENT $q M=$M WGS=$w pass=$p (bench_frontend_tinym, n=100, GPU0)" >> "$LOG"
  env DG_FP4_MATH_WGS=$w timeout 600 $TR --standalone --nproc_per_node=8 tests/bench_frontend_tinym.py \
    --quant $q --global-tokens $M --iters 100 > "$OUTD/event_${q}_M${M}_wg${w}_p${p}.log" 2>&1
  echo "EXIT=$?" >> "$LOG"
  grep -E "^ *(FE|Mega|FE\+Mega):" "$OUTD/event_${q}_M${M}_wg${w}_p${p}.log" >> "$LOG"
}
if [ "$MODE" = event ] || [ "$MODE" = all ]; then
  for q in mxfp4 qoq; do for M in 8 16; do for w in 2 3; do event_one $q $M $w 1; done; done; done
  for q in qoq mxfp4; do for M in 16 8; do for w in 3 2; do event_one $q $M $w 2; done; done; done
  echo ALL_EVENT_DONE >> "$LOG"
fi

if [ "$MODE" = nsys ] || [ "$MODE" = all ]; then
  for w in 2 3; do
    wait_idle || exit 1
    echo "--- NSYS customer method WGS=$w TOKENS 2 4 8 16 e2e+mega fused" >> "$LOG"
    env DG_FP4_MATH_WGS=$w TOKENS_LIST="2 4 8 16" SCOPES="e2e mega" BACKENDS=fused FORCE=1 \
      OUT="$OUTD/nsys_wg$w" timeout 3000 bash scripts/capture_four_api_h20_timelines.sh > "$OUTD/nsys_wg$w.log" 2>&1
    echo "EXIT=$?" >> "$LOG"
    tail -3 "$OUTD/nsys_wg$w.log" >> "$LOG"
  done
  python3 scripts/summarize_knob_captures.py --knob 2 "$OUTD/nsys_wg2" --knob 3 "$OUTD/nsys_wg3" --fused-only > "$OUTD/nsys_summary.txt" 2>&1
  cat "$OUTD/nsys_summary.txt" >> "$LOG"
  echo ALL_NSYS_DONE >> "$LOG"
fi
echo "ALL_DONE $MODE" >> "$LOG"
