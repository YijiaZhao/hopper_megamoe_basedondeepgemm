#!/bin/bash
# Round-5 single-GPU chain: SASS sizes, bit-identity gates (lean, pruned, lean+pruned vs baseline), standalone A/B.
# Usage (in four_api_build, repo root): FE5_GPU=7 bash tests/fe5_chain.sh [outdir]
cd "$(dirname "$0")/.."
OUT=${1:-/raid/kimi/results/fe5}; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${FE5_GPU:-7} DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc
echo "CHAIN_START $(date +%T) $(git log --oneline -1)"
SO=$(readlink -f deep_gemm/_C*.so)
cuobjdump -sass "$SO" > "$OUT/v1.sass" 2>&1
python3 - "$OUT/v1.sass" <<'EOF'
import re, sys
cur = None; n = {}
for line in open(sys.argv[1]):
    m = re.match(r"\s*Function : (\S+)", line)
    if m: cur = m.group(1); n[cur] = 0; continue
    if cur and re.match(r"\s*/\*[0-9a-f]{4,}\*/", line): n[cur] += 1
for k, v in n.items():
    if "router_cc_lean_kernel" in k or "router_quant_topk_kernelILi1ELi0ELb1ELb1ELb0ELi0ELi44" in k or "router_quant_topk_kernelILi1ELi1ELb1ELb1ELb0ELi0ELi44" in k:
        print(f"SASS {v} instructions ({v * 16 // 1024} KB) {k[:120]}")
EOF
echo "--- identity gates (64 seeds x rows 1,2 x both quants; knob-0 topk + knob-1 keys + x/x_sf)"
timeout 600 python3 tests/fe5_ident.py --save "$OUT/ident_base.pt" 2>&1 | grep FE5_IDENT
DG_FE_CC_LEAN=1 timeout 600 python3 tests/fe5_ident.py --ref "$OUT/ident_base.pt" 2>&1 | grep -E "FE5_IDENT|Error|error"
DG_FE_CC_SELECT=pruned timeout 600 python3 tests/fe5_ident.py --ref "$OUT/ident_base.pt" 2>&1 | grep -E "FE5_IDENT|Error|error"
DG_FE_CC_LEAN=1 DG_FE_CC_SELECT=pruned timeout 600 python3 tests/fe5_ident.py --ref "$OUT/ident_base.pt" 2>&1 | grep -E "FE5_IDENT|Error|error"
echo "--- standalone A/B"
bash tests/fe5_ab.sh 200 "$OUT/ab"
echo "CHAIN_DONE $(date +%T)"
