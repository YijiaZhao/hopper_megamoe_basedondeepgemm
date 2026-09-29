#!/bin/bash
# 8-rank E2E (frontend + forced-balanced override + MegaMoE in one graph): $1 = fp8|mxfp4|nvfp4, $2 = tree (DeepGEMM)
D=/lustre/fsw/portfolios/coreai/users/kimiz/megamoe_b200
T=$D/${2:-DeepGEMM}
cd $T
export PYTHONPATH=$T DG_JIT_CACHE_DIR=$D/jit_cache CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=10.0 TORCH_EXTENSIONS_DIR=$D/fe_sm100/fe_build
pkill -9 -f spawn_mai[n]; sleep 1
run(){ echo "### E2E $1"; timeout 1200 python $D/fe_sm100/test_e2e_b200.py --num-processes 8 --num-experts 384 --hidden 3072 --intermediate-hidden 1280 --num-topk 8 --num-max-tokens-per-rank 128 --num-correctness-tests 0 --balanced --torch-ref --bench-stream 30 --act-format $2 $3 2>&1 | grep -E "rel_rmse per|E2E us|KSPAN us|FE us|GAP us|Traceback|rror|assert" | grep -v "multicast\|Wno" | head -12 | cut -c1-300; pkill -9 -f spawn_mai[n]; }
run "$1 M2"  $1 "--num-tokens 1 --active-ranks 0,4"
run "$1 M4"  $1 "--num-tokens 1 --active-ranks 0,1,4,5"
run "$1 M8"  $1 "--num-tokens 1"
run "$1 M16" $1 "--num-tokens 2"
echo RUN_DONE
