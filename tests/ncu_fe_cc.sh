#!/usr/bin/env bash
# NCU of the cc router (DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc, defaults) on ONE GPU, rows 1 mxfp4 + rows 2 qoq.
# --set full --clock-control none; default cache control (L2 flushed between replay passes) for the main
# report, plus one --cache-control none run per cell (L2 persist verification: weight-load L2 hit rate).
# Usage (in four_api_build, repo root): NCU_FE_GPU=6 bash tests/ncu_fe_cc.sh [outdir]
set -uo pipefail
OUT=${1:-/raid/kimi/results/fe_cc4/ncu_fe_cc}; GPU=${NCU_FE_GPU:-6}; WARM=${NCU_FE_WARMUP:-5}
mkdir -p "$OUT"; export CUDA_VISIBLE_DEVICES=$GPU
export DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc DG_FE_ROUTER_L2_PERSIST_VERBOSE=1
nvidia-smi -i "$GPU" --query-gpu=index,name,clocks.sm,clocks.mem,memory.total --format=csv > "$OUT/gpu.txt"
python3 -c "import torch; d=torch.cuda.current_device(); print('MaxAccessPolicyWindowSize', torch.cuda.get_device_properties(d)); import ctypes" >> "$OUT/gpu.txt" 2>&1
run_one() {  # quant rows cachectl tag
  echo "=== $4 ($(date +%T))"
  timeout 900 ncu --target-processes application-only --kernel-name regex:router_quant_topk_kernel --launch-skip "$WARM" --launch-count 1 \
    --set full --import-source no --clock-control none --cache-control "$3" -f -o "$OUT/$4" \
    python3 tests/ncu_frontend_tinym.py --quant "$1" --rows "$2" --tinym 1 --warmup "$WARM" --grid auto --mma cc --l2-flush 1 > "$OUT/$4.run.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$4.run.log"
  if [ -f "$OUT/$4.ncu-rep" ]; then
    ncu --import "$OUT/$4.ncu-rep" --page details > "$OUT/$4.details.txt" 2>&1
    ncu --import "$OUT/$4.ncu-rep" --page details --csv > "$OUT/$4.details.csv" 2>&1
    ncu --import "$OUT/$4.ncu-rep" --page raw --csv > "$OUT/$4.raw.csv" 2>&1
  fi
}
run_one mxfp4 1 all  fe_cc_rows1_mxfp4
run_one qoq   2 all  fe_cc_rows2_qoq
run_one mxfp4 1 none fe_cc_rows1_mxfp4_nocachectl
run_one qoq   2 none fe_cc_rows2_qoq_nocachectl
echo "NCU_ALL_DONE $(date +%T)"
