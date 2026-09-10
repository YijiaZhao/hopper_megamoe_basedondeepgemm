#!/usr/bin/env bash
# Validate DG_FP4_FUSE_L1L2 (two-layer fusion for tiny M, docs/fuse_l1l2_design.md):
#   corr   : mxfp4 T=2/8/8/8/16/16 (fused) + 128/512 (untouched path), qoq T=8/16 (fused)
#   probe  : phase-stamp probe + per-CTA task log (PROBE_TASKLOG=1), knob 1 vs 0, mxfp4 2/8/16 + qoq 8
#   stress : 200-iteration graph-replay probe at M=8 (mxfp4 + qoq), knob 1
#   perf   : skew-free official-method nsys A/B (scripts/run_knob_nsys_ab.sh), 2 passes
# Usage (inside four_api_build): bash scripts/run_fuse_validate.sh [corr|probe|stress|perf|all] [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}; TAG=${2:-}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
TR=(timeout 600 /usr/local/bin/torchrun --standalone --nproc_per_node=8)
LOG=corr_fuse$TAG.log

run_corr() {  # $1 quant api, $2 tokens, $3.. env
  local api=$1 T=$2; shift 2
  echo "--- $api T=$T ($*)" >> "$LOG"
  env "$@" "${TR[@]}" tests/test_four_api_correctness.py --apis "$api" --tokens "$T" >> "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  : > "$LOG"
  for T in 2 8 8 8 16 16 128 512; do
    run_corr mxfp4_mega_moe_fused "$T" DG_FP4_FUSE_L1L2=1 DG_FP4_SPIN_TIMEOUT=1
  done
  for T in 8 16; do
    run_corr qoq_mega_moe_fused "$T" DG_FP4_FUSE_L1L2=1 DG_FP4_SPIN_TIMEOUT=1
  done
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = probe ] || [ "$MODE" = all ]; then
  for it in mxfp4:8 mxfp4:2 mxfp4:16 qoq:8; do
    q=${it%%:*}; m=${it##*:}
    for v in 1 0; do
      PROBE_TASKLOG=1 DG_FP4_FUSE_L1L2=$v "${TR[@]}" tests/profile_fused_phase_stamps.py \
        --quant "$q" --global-tokens "$m" --iters 20 > "probe_${q}_m${m}_fuse${v}${TAG}.log" 2>&1
      echo "PROBE_EXIT=$?" >> "probe_${q}_m${m}_fuse${v}${TAG}.log"
    done
  done
  echo ALL_PROBES_DONE >> "$LOG"
fi

if [ "$MODE" = stress ] || [ "$MODE" = all ]; then
  for q in mxfp4 qoq; do
    DG_FP4_FUSE_L1L2=1 DG_FP4_SPIN_TIMEOUT=1 "${TR[@]}" tests/profile_fused_phase_stamps.py \
      --quant "$q" --global-tokens 8 --iters 200 > "probe_${q}_m8_fuse1_stress$TAG.log" 2>&1
    echo "STRESS_${q}_EXIT=$?" >> "$LOG"
  done
  echo ALL_STRESS_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  KNOB=DG_FP4_FUSE_L1L2 OFF=0 ON=1 PASSES=${PASSES:-2} bash scripts/run_knob_nsys_ab.sh \
    "/raid/kimi/results/fuse_l1l2_ab$TAG" mxfp4:8 mxfp4:2 mxfp4:16 qoq:8 qoq:16 > "perf_fuse$TAG.log" 2>&1
  echo "PERF_EXIT=$?" >> "$LOG"
  echo ALL_PERF_DONE >> "$LOG"
fi
