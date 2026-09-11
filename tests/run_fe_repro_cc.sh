#!/bin/bash
# 10 fresh processes x {rows 1,2} x {mxfp4,qoq}, cc router defaults. Usage (in four_api_build, repo root):
#   CUDA_VISIBLE_DEVICES=k bash tests/run_fe_repro_cc.sh [procs=10] [iters=100] [tag] [gap=clone|compute|sleep]
# gap: what the host does between two stamped launches (clone = back-to-back, only the stamp clone; compute = per-launch
# D2H reduction, ms-scale gaps; sleep = 5 ms sleep). DG_FE_SELECT_IN_MEGA=1 measures the keys-only FE (kernel end = last keys written).
set -u
cd "$(dirname "$0")/.."
P=${1:-10}; N=${2:-100}; TAG=${3:-run}; GAP=${4:-clone}; SEL=${DG_FE_SELECT_IN_MEGA:-0}
echo "REPRO_START $(date +%T) gpu=$CUDA_VISIBLE_DEVICES clocks=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader -i 0 2>/dev/null | head -1)"
for p in $(seq 1 $P); do for rows in 1 2; do for q in mxfp4 qoq; do
  DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc timeout 900 python3 tests/fe_repro_cc.py --quant $q --rows $rows --iters $N --gap $GAP --tag "$TAG/p$p" 2>&1 | grep -E "REPRO|Error|error"
done; done; done
echo "REPRO_DONE $(date +%T)"
