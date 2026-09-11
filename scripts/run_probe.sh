#!/usr/bin/env bash
# Run the fused phase-stamp probe for a list of global-token sizes.
# Usage (inside four_api_build container): bash scripts/run_probe.sh [quant] M1 M2 ...
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
QUANT=${1:-mxfp4}; shift || true
[ $# -gt 0 ] || set -- 2 8 16
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
nvidia-smi -L | head -1
for M in "$@"; do
  log="probe_${QUANT}_m${M}${LOG_TAG:-}.log"
  torchrun --standalone --nproc_per_node=8 tests/profile_fused_phase_stamps.py ${PROBE_ARGS:-} \
    --quant "$QUANT" --global-tokens "$M" --iters 20 > "$log" 2>&1
  echo "PROBE_EXIT=$?" >> "$log"
done
echo ALL_PROBES_DONE
