#!/usr/bin/env bash
# Validate the BM8 RF kKBlocksPerStage knob (DG_FP4_KBLOCKS_PER_STAGE, default KB=4):
# correctness (mxfp4 fused T=2/8/8/16/128/512 + qoq T=8) with ptxas -v, then probes
# (KB twice, 2-block control once). Usage (in four_api_build): bash scripts/run_k4_validate.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
KB=${KB:-4}
export DG_FP4_KBLOCKS_PER_STAGE=$KB
LOG="k4_validate_kb${KB}${LOG_TAG:-}.log"
: > "$LOG"
run() {
  echo "### $*" >> "$LOG"
  DG_JIT_PTXAS_VERBOSE=1 /usr/local/bin/torchrun --standalone --nproc_per_node=8 "$@" >> "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
}
for T in 2 8 8 16 128 512; do
  run tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused --tokens "$T"
done
run tests/test_four_api_correctness.py --apis qoq_mega_moe_fused --tokens 8
if [ "${SKIP_PROBE:-0}" != 1 ]; then
  LOG_TAG="_kb${KB}_a" bash scripts/run_probe.sh mxfp4 2 8 16 >> "$LOG" 2>&1
  LOG_TAG="_kb${KB}_b" bash scripts/run_probe.sh mxfp4 2 8 16 >> "$LOG" 2>&1
  DG_FP4_KBLOCKS_PER_STAGE=2 LOG_TAG="_kb2_ctl" bash scripts/run_probe.sh mxfp4 2 8 16 >> "$LOG" 2>&1
fi
echo ALL_DONE >> "$LOG"
