#!/bin/bash
# tc16 follow-ups on the GPU host (inside four_api_build, --privileged for ncu):
#   1. one NCU --set full (clock-control none) of the tc16 FE kernel, rows 1 mxfp4, on $NCU_FE_GPU (default 7);
#   2. the 8-rank fused correctness gate (tests/run_ccrouter_corr8.sh) with DG_FE_TINYM_MMA=tc16, skipped when
#      another agent holds /raid/kimi/results/CAPTURE_MERGE_RUNNING.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=${NCU_FE_OUT:-/raid/kimi/results/ncu_fe}; mkdir -p "$OUT"
GPU=${NCU_FE_GPU:-7}; MMA=${DG_FE_TINYM_MMA:-tc16}
tag="fe_${MMA}_M2_mxfp4"
echo "=== ncu $tag ($(date +%T))"
CUDA_VISIBLE_DEVICES=$GPU DG_FE_TINYM=1 DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=$MMA timeout 900 ncu --target-processes application-only \
  --kernel-name regex:router_quant_topk_kernel --launch-skip 5 --launch-count 1 \
  --set full --import-source no --clock-control none -f -o "$OUT/$tag" \
  python3 tests/ncu_frontend_tinym.py --quant mxfp4 --rows 1 --tinym 1 --warmup 5 --grid auto --mma $MMA > "$OUT/$tag.run.log" 2>&1
echo "ncu EXIT=$?"
if [ -f "$OUT/$tag.ncu-rep" ]; then
  ncu --import "$OUT/$tag.ncu-rep" --page details > "$OUT/$tag.details.txt" 2>&1
  grep -E "Duration|Registers Per Thread|Achieved Occupancy|Theoretical Occupancy|Memory Throughput|L2 Hit Rate|L1/TEX Hit|DRAM Throughput|Compute \(SM\) Throughput|Block Size|Grid Size|Executed Ipc Active|Issue Slots Busy|No Eligible|Eligible Warps|Active Warps Per Scheduler|Warp Cycles Per Issued|Stall Long Scoreboard|Waves Per SM|Shared Memory Configuration Size|Dynamic Shared|Static Shared" "$OUT/$tag.details.txt" | head -60
fi
if [ -e /raid/kimi/results/CAPTURE_MERGE_RUNNING ]; then echo "corr8 SKIPPED: CAPTURE_MERGE_RUNNING present"; exit 0; fi
echo "=== corr8 mma=$MMA ($(date +%T))"
DG_FE_TINYM_MMA=$MMA bash tests/run_ccrouter_corr8.sh
echo "NCU_CORR8 DONE $(date +%T)"
