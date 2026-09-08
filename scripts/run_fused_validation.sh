#!/usr/bin/env bash
# Fused MegaMoE validation driver (inside the four_api_build container):
# waits for idle GPUs, runs the mxfp4/qoq fused correctness matrix, then the
# phase-stamp perf probe. Usage: LOG_TAG=_dense bash scripts/run_fused_validation.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT" CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1
TAG=${LOG_TAG:-}
SUMMARY="validation${TAG}.summary"
: > "$SUMMARY"
wait_idle() {
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; do
    echo "$(date +%T) GPUs busy, waiting" >> "$SUMMARY"; sleep 20
  done
}
run_case() {  # api tokens
  wait_idle
  local log="correctness_${1}_t${2}${TAG}.log"
  torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis "$1" --tokens "$2" > "$log" 2>&1
  echo "CORRECTNESS api=$1 tokens=$2 EXIT=$?" >> "$SUMMARY"
}
for T in 2 8 16 128 512; do run_case mxfp4_mega_moe_fused "$T"; done
run_case qoq_mega_moe_fused 8
wait_idle
LOG_TAG="$TAG" bash scripts/run_probe.sh mxfp4 2 8 16 >> "$SUMMARY" 2>&1
echo VALIDATION_DONE >> "$SUMMARY"
