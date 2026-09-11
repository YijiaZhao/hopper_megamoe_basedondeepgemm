#!/bin/bash
# 8-rank correctness gate for the cc router: tests/test_four_api_correctness.py (fused mxfp4 + qoq)
# with DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc for per-rank tokens 1, 2 (cc range) and 8 (WMMA fallback).
# Usage (inside four_api_build, all 8 GPUs idle): bash tests/run_ccrouter_corr8.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DG_FE_TINYM=1 DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=${DG_FE_TINYM_MMA:-cc}
for tok in 1 2 8; do
  echo "### corr8 mma=$DG_FE_TINYM_MMA merge=${DG_FE_CC_MERGE:-poll} tokens/rank=$tok"
  timeout 900 /usr/local/bin/torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
      --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens $tok 2>&1 | grep -E "cos|PASS|FAIL|Error|error|Traceback|exitcode" | head -20
done
