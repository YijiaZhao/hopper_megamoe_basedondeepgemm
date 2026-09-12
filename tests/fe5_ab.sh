#!/bin/bash
# Round-5 standalone A/B of the cc router knobs on ONE GPU: baseline vs DG_FE_CC_LEAN=1 (and, knob-0 path,
# DG_FE_CC_SELECT=pruned). Each cell = one fresh process of tests/fe_repro_cc.py (back-to-back stamped
# launches after a 256 MB L2 flush) under nsys; reports the nsys router-kernel span (avg / med / min over
# the launches, the same GPU-span source as the customer method) and the stamp chain kernel end.
# Usage (in four_api_build, repo root): FE5_GPU=7 bash tests/fe5_ab.sh [iters=200] [outdir]
cd "$(dirname "$0")/.."
N=${1:-200}; OUT=${2:-/raid/kimi/results/fe5/ab}; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${FE5_GPU:-7} DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc
echo "AB_START $(date +%T) $(git log --oneline -1)"
run_cell() {   # selmega lean sel quant rows
  local tag="sm$1_lean$2_$3_$4_r$5"
  DG_FE_SELECT_IN_MEGA=$1 DG_FE_CC_LEAN=$2 DG_FE_CC_SELECT=$3 timeout 600 nsys profile -t cuda -s none --cpuctxsw=none -f true -o "$OUT/$tag" \
    python3 tests/fe_repro_cc.py --quant "$4" --rows "$5" --iters "$N" --gap clone --tag "$tag" > "$OUT/$tag.log" 2>&1
  local repro; repro=$(grep "^REPRO" "$OUT/$tag.log" | sed 's/.*end\[med min p90\]= \([0-9. ]*\) |.*/\1/')
  local span; span=$(nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/$tag.nsys-rep" 2>/dev/null | grep -E "router_(quant_topk|cc_lean)_kernel" | head -1 \
    | awk -F, '{printf "n=%s avg=%.2f med=%.2f min=%.2f max=%.2f", $3, $4/1000, $5/1000, $6/1000, $7/1000}')
  echo "AB selmega=$1 lean=$2 sel=$3 quant=$4 rows=$5 | nsys_span_us: $span | stamp_end[med min p90]= $repro"
}
for cell in "mxfp4 1" "qoq 1" "mxfp4 2" "qoq 2"; do
  set -- $cell
  run_cell 1 0 insert $1 $2
  run_cell 1 1 insert $1 $2
  run_cell 0 0 insert $1 $2
  run_cell 0 0 pruned $1 $2
  run_cell 0 1 insert $1 $2
  run_cell 0 1 pruned $1 $2
done
echo "AB_DONE $(date +%T)"
