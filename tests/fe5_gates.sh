#!/bin/bash
# 8-rank correctness gates for the Fable frontend + fused Mega, library defaults (rows <= 2 per rank = the cc
# router (T=1, 2), rows > 2 = swapab):
#   * tests/test_select_in_mega.py: select-in-Mega topk / y equality vs the select-in-FE path, both quants, M 2 / 16, 50 seeds
#   * tests/test_four_api_correctness.py --frontend fe --router-ref torch: FE + fused Mega vs a pure-torch
#     reference (bf16 router GEMM -> top-8 -> softmax -> torch per-token cast -> dequantised expert GEMMs ->
#     weighted combine), T = 1 2 8 16 32 per rank, both fused APIs; T=32 also with --slot-check
#   * tests/test_four_api_correctness.py --tokens 32 --hot-rows 12 --slot-check (balanced routing, second BM8 block)
# Usage (in four_api_build, repo root): bash tests/fe5_gates.sh
cd "$(dirname "$0")/.."
echo "GATES_START $(date +%T) $(git log --oneline -1)"
TR="timeout 900 /usr/local/bin/torchrun --standalone --nproc_per_node=8"
for q in mxfp4 qoq; do for m in 2 16; do
  echo "### select_in_mega quant=$q M=$m"
  $TR tests/test_select_in_mega.py --quant $q --global-tokens $m --seeds 50 2>&1 | grep -E "SELMEGA_GATE|Error|error|Traceback" | head -5
done; done
for t in 1 2 8 16 32; do
  echo "### four_api_correctness --frontend fe --router-ref torch tokens/rank=$t (M=$((t * 8)))"
  $TR tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --router-ref torch --tokens $t 2>&1 \
    | grep -E "ROUTER_REF|RESULT|SLOT_CHECK|Error|error|Traceback|assert" | head -12
done
echo "### four_api_correctness --frontend fe --slot-check tokens 32"
$TR tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --router-ref torch --tokens 32 --slot-check 2>&1 | grep -E "ROUTER_REF|RESULT|SLOT_CHECK|Error|Traceback" | head -12
echo "### four_api_correctness balanced --tokens 32 --hot-rows 12 --slot-check"
$TR tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens 32 --hot-rows 12 --slot-check 2>&1 | grep -E "RESULT|SLOT_CHECK|Error|Traceback" | head -12
echo "GATES_DONE $(date +%T)"
