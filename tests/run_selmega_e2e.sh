#!/bin/bash
# 8-rank FE+Mega verdict for DG_FE_SELECT_IN_MEGA (cc router): knob 0 vs 1 x {mxfp4,qoq} x M {2,8,16},
# tests/bench_frontend_tinym.py (FE / Mega / FE+Mega graph CUDA-event times, n=100, host barrier), then the
# 8-rank correctness gate (tests/test_four_api_correctness.py, fused api) with the knob on. Needs all 8 GPUs.
# Usage (in four_api_build, repo root): bash tests/run_selmega_e2e.sh [iters=100] [knobs="0 1"] [quants="mxfp4 qoq"] [Ms="2 8 16"]
cd "$(dirname "$0")/.."
N=${1:-100}; KNOBS=${2:-"0 1"}; QS=${3:-"mxfp4 qoq"}; MS=${4:-"2 8 16"}
export DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc DG_FE_STAMPS=1
echo "E2E_START $(date +%T)"
for k in $KNOBS; do for q in $QS; do for m in $MS; do
  echo "### knob=$k quant=$q M=$m"
  DG_FE_SELECT_IN_MEGA=$k timeout 900 /usr/local/bin/torchrun --standalone --nproc_per_node=8 tests/bench_frontend_tinym.py --quant $q --global-tokens $m --iters $N 2>&1 \
    | grep -E "^== frontend|^ +(FE|Mega|FE\+Mega|FE\+Mega/st):|topk written|keys written|merger|Error|error|Traceback" | head -40
done; done; done
for q in $QS; do for m in 2 16; do
  echo "### gate quant=$q M=$m (knob 1 vs knob 0 topk equality + y cos, 8 ranks, 50 seeds)"
  timeout 900 /usr/local/bin/torchrun --standalone --nproc_per_node=8 tests/test_select_in_mega.py --quant $q --global-tokens $m --seeds 50 2>&1 | grep -E "SELMEGA_GATE|Error|error|Traceback" | head -8
done; done
echo "E2E_DONE $(date +%T)"
