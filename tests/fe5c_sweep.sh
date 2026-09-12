#!/bin/bash
# FE + Mega end-to-end correctness sweep vs the pure-torch MoE reference (tests/test_four_api_correctness.py
# --reference torch-moe: bf16 router GEMM -> top-8 -> softmax -> torch per-token cast -> dequantised expert GEMMs ->
# weighted combine; REAL routing), 8 ranks, both fused quants per launch.
#   shapes: customer M = 2 / 4 (owner layout, --tokens 1 --global-tokens M), 8 (--tokens 1), 16 (--tokens 2) and
#           T = 8 / 16 / 32 tokens per rank (M = 64 / 128 / 256); T = 2 is the M = 16 cell
#   axes:   SELS (DG_FE_SELECT_IN_MEGA; 1 = the pipeline configuration, effective for <= 2 rows per rank) x
#           ROUTINGS (normal = the FE's real top-8, balanced = DG_FE_FORCE_BALANCED=1) x SEEDS
#   extras: --tokens 32 --hot-rows 12 --slot-check (synthetic balanced, second BM8 pool block) and
#           --frontend fe --reference torch-moe --tokens 32 --slot-check
# Usage (repo root, 8 GPUs): SELS="1 0" ROUTINGS="normal balanced" SEEDS="20260903" NSEEDS=1 OUT=/raid/kimi/results/fe5c bash tests/fe5c_sweep.sh
# Summarise: python3 scripts/summarize_fe5c_sweep.py $OUT/*.log
cd "$(dirname "$0")/.."
OUT=${OUT:-/tmp/fe5c_sweep}; mkdir -p "$OUT"
SELS=${SELS:-"1 0"}; ROUTINGS=${ROUTINGS:-"normal balanced"}; SEEDS=${SEEDS:-"20260903"}
NSEEDS=${NSEEDS:-1}   # seeds per launch (--seeds: seed .. seed+N-1 in one process)
SHAPES=${SHAPES:-"M2:--tokens_1_--global-tokens_2 M4:--tokens_1_--global-tokens_4 M8:--tokens_1 M16:--tokens_2 T8:--tokens_8 T16:--tokens_16 T32:--tokens_32"}
EXTRAS=${EXTRAS:-1}
TR=${TR:-"timeout 900 torchrun --standalone --nproc_per_node=8"}
APIS="--apis mxfp4_mega_moe_fused qoq_mega_moe_fused"
GREP="ROUTING|ROUTER_REF|RESULT|SUMMARY|SEED_FAIL|SELECT_IN_MEGA|SLOT_CHECK|Traceback|Error|assert"
echo "SWEEP_START $(date +%T) $(git log --oneline -1) host=$(hostname) SELS=[$SELS] ROUTINGS=[$ROUTINGS] SEEDS=[$SEEDS]"
for sel in $SELS; do for routing in $ROUTINGS; do for seed in $SEEDS; do for shape in $SHAPES; do
  name=${shape%%:*}; flags=${shape#*:}; flags=${flags//_/ }
  fb=0; [ "$routing" = balanced ] && fb=1
  tag="sel${sel}_${routing}_seed${seed}x${NSEEDS}_${name}"
  log="$OUT/$tag.log"
  if grep -q "^RESULT api=qoq" "$log" 2>/dev/null; then echo "skip $tag (done)"; continue; fi
  echo "### $tag  $(date +%T)"
  DG_FE_SELECT_IN_MEGA=$sel DG_FE_FORCE_BALANCED=$fb \
    $TR tests/test_four_api_correctness.py $APIS --frontend fe --reference torch-moe --seed "$seed" --seeds "$NSEEDS" $flags > "$log" 2>&1
  echo "rc=$?"; grep -E "$GREP" "$log" | head -14
done; done; done; done
if [ "$EXTRAS" = 1 ]; then
  tag="extra_hotrows12_slotcheck"; log="$OUT/$tag.log"
  echo "### $tag  $(date +%T)"
  $TR tests/test_four_api_correctness.py $APIS --tokens 32 --hot-rows 12 --slot-check > "$log" 2>&1
  echo "rc=$?"; grep -E "$GREP" "$log" | head -8
  tag="extra_fe_T32_slotcheck"; log="$OUT/$tag.log"
  echo "### $tag  $(date +%T)"
  $TR tests/test_four_api_correctness.py $APIS --frontend fe --reference torch-moe --tokens 32 --slot-check > "$log" 2>&1
  echo "rc=$?"; grep -E "$GREP" "$log" | head -8
fi
echo "SWEEP_DONE $(date +%T)"
