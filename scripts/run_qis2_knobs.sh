#!/usr/bin/env bash
# QoQ inline-s2 knob matrix (DG_FP4_QIS2_PREFETCH_PACKED x DG_FP4_QIS2_RAWU8):
# correctness (bit-exact cos_min vs the fused decode), ptxas -v / SASS pipe check of
# the default build, and the run_probe.sh perf table. Inside four_api_build:
#   MODE=corr|perf|sass|all bash scripts/run_qis2_knobs.sh
# GPU discipline: every run waits for an idle GPU, no live nvcc and no
# /raid/kimi/results/OFFICIAL_R4_RUNNING flag.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-all}
TAG=${TAG:-}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
TR=/usr/local/bin/torchrun
CACHE=${DG_JIT_CACHE_DIR:-/raid/kimi/dg_qoq2/.jit_qis2$TAG}
export DG_JIT_CACHE_DIR="$CACHE"
LOG=qis2_knobs$TAG.log

wait_idle() {
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    m=$(ps -C nvcc -o stat= 2>/dev/null | grep -v '^Z' | grep -c .)
    [ "$n" = 0 ] && [ "$m" = 0 ] && [ ! -e /raid/kimi/results/OFFICIAL_R4_RUNNING ] && break
    sleep 5
  done
}

run_corr() {  # $1 = api, $2 = T, $3.. = env assignments
  local api=$1 T=$2; shift 2
  wait_idle
  echo "--- CORR $api T=$T ($*)" >> "$LOG"
  env "$@" timeout 300 $TR --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis "$api" --tokens "$T" > corr_tmp.log 2>&1
  local rc=$?
  grep -h "ptxas info\|bytes spill\|bytes stack\|Used [0-9]* registers\|cos_min\|Error\|error" corr_tmp.log | grep -v "^$" | sort -u >> "$LOG"
  echo "EXIT=$rc" >> "$LOG"
}

run_probe() {  # $1 = tag, $2 = M list, $3.. = env assignments
  local tag=$1 ms=$2; shift 2
  for M in $ms; do
    wait_idle
    env "$@" LOG_TAG="_$tag$TAG" timeout 300 bash scripts/run_probe.sh qoq "$M" > /dev/null 2>&1
    echo "--- PROBE qoq M=$M ($tag: $*)" >> "$LOG"
    sed -n '/=== fused/,$p' "probe_qoq_m${M}_$tag$TAG.log" | grep -v "PROBE_ITER\|Warning\|warn" >> "$LOG"
  done
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  : > "$LOG"
  # default build (1,1) with ptxas -v: the first run JIT-compiles the kernel
  run_corr qoq_mega_moe_fused 2 DG_JIT_PTXAS_VERBOSE=1 DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=1
  for T in 8 8 16; do
    run_corr qoq_mega_moe_fused $T DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=1
  done
  run_corr qoq_mega_moe_fused 8 DG_JIT_PTXAS_VERBOSE=1 DG_FP4_QIS2_PREFETCH_PACKED=0 DG_FP4_QIS2_RAWU8=0
  run_corr qoq_mega_moe_fused 8 DG_JIT_PTXAS_VERBOSE=1 DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=0
  run_corr qoq_mega_moe_fused 8 DG_JIT_PTXAS_VERBOSE=1 DG_FP4_QIS2_PREFETCH_PACKED=0 DG_FP4_QIS2_RAWU8=1
  run_corr mxfp4_mega_moe_fused 8
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = sass ] || [ "$MODE" = all ]; then
  echo "--- SASS (cubins under $CACHE)" >> "$LOG"
  for cb in $(find "$CACHE" -name '*.cubin' | grep -i qoq); do
    echo "cubin: $cb" >> "$LOG"
    cuobjdump -sass "$cb" > sass_tmp.txt 2>/dev/null
    echo "IGMMA=$(grep -c 'IGMMA' sass_tmp.txt) HGMMA=$(grep -c 'HGMMA' sass_tmp.txt) DEPBAR=$(grep -c 'WARPGROUP.DEPBAR' sass_tmp.txt) ARRIVE=$(grep -c 'WARPGROUP.ARRIVE' sass_tmp.txt) CALL=$(grep -c ' CALL' sass_tmp.txt) LDL=$(grep -c ' LDL' sass_tmp.txt) STL=$(grep -c ' STL' sass_tmp.txt) IDP=$(grep -c ' IDP' sass_tmp.txt) BAR=$(grep -c ' BAR.SYNC' sass_tmp.txt)" >> "$LOG"
    # longest run of consecutive IGMMA lines (no DEPBAR between them)
    grep -E 'IGMMA|WARPGROUP' sass_tmp.txt | awk '{for(i=1;i<=NF;i++) if($i ~ /IGMMA|WARPGROUP/){print $i; break}}' | awk 'BEGIN{r=0;m=0} /IGMMA/{r++; if(r>m)m=r; next} {r=0} END{print "max_consecutive_IGMMA=" m}' >> "$LOG"
    grep -E 'IGMMA|WARPGROUP' sass_tmp.txt | awk '{for(i=1;i<=NF;i++) if($i ~ /IGMMA|WARPGROUP/){print $i; break}}' | uniq -c | head -60 >> "$LOG"
  done
  echo ALL_SASS_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  run_probe pf0raw0 "8 16" DG_FP4_QIS2_PREFETCH_PACKED=0 DG_FP4_QIS2_RAWU8=0
  run_probe pf1raw0 "8 16" DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=0
  run_probe pf0raw1 "8 16" DG_FP4_QIS2_PREFETCH_PACKED=0 DG_FP4_QIS2_RAWU8=1
  run_probe pf1raw1 "8 16" DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=1
  run_probe pf1raw1 "2" DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=1
  echo ALL_PERF_DONE >> "$LOG"
fi
if [ "$MODE" = perf2 ]; then
  # second pass, reverse order (run-to-run drift check)
  run_probe pf1raw1 "16 8" DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=1
  run_probe pf0raw1 "16 8" DG_FP4_QIS2_PREFETCH_PACKED=0 DG_FP4_QIS2_RAWU8=1
  run_probe pf1raw0 "16 8" DG_FP4_QIS2_PREFETCH_PACKED=1 DG_FP4_QIS2_RAWU8=0
  run_probe pf0raw0 "16 8" DG_FP4_QIS2_PREFETCH_PACKED=0 DG_FP4_QIS2_RAWU8=0
  echo ALL_PERF2_DONE >> "$LOG"
fi
echo ALL_DONE >> "$LOG"
