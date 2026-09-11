#!/usr/bin/env bash
# FE equality / bit-identity of the perf/merge-fe tip on a ComputeLab 4 x H20 node (correctness only,
# no clock lock). Runs inside lmsysorg/sglang:dev: clone -> build -> 4 GPUs in parallel:
#   GPU0 tests/test_frontend_fe78.py --mma auto   (library defaults: cc + auto grid + L2 persist for
#        rows 1/2, swapab + fragment on the 96 grid for rows 8/16) 1000 seeds x rows {1,2,8,16} x 2 quants
#   GPU1 --grid auto --mma cc (explicit; rows 8/16 fall back to full-K WMMA)          1000 seeds
#   GPU2 --grid 96 --mma swapab --wlayout fragment (explicit)                          1000 seeds
#   GPU3 tests/test_frontend_tinym.py bit-identity: legacy tiny path (96 wmma) 200 seeds, then the
#        library defaults (expect topk_weights bf16 flips only on rows 1/2), then defaults + persist
#   docker run -d --gpus all --ipc=host -v <out>:/out lmsysorg/sglang:dev bash -c \
#     "git clone -b perf/merge-fe https://github.com/YijiaZhao/hopper_megamoe_basedondeepgemm.git /work/repo && bash /work/repo/scripts/run_merge_equality_computelab.sh"
set -uo pipefail
OUT=${OUT:-/out}; mkdir -p "$OUT"; ROOT=${ROOT:-/work/repo}; cd "$ROOT" || exit 1
SEEDS=${SEEDS:-1000}
export PYTHONPATH="$ROOT" CUDA_HOME=/usr/local/cuda PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST=9.0a
export PATH="/usr/local/cuda/bin:$PATH" DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export DG_JIT_CACHE_DIR="$OUT/jit"
echo "=== computelab merge equality $(git rev-parse --short HEAD) $(date)" > "$OUT/STATUS.log"
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm --format=csv >> "$OUT/STATUS.log"
git submodule update --init --depth 1 third-party/cutlass third-party/fmt >> "$OUT/build.log" 2>&1
bash develop.sh >> "$OUT/build.log" 2>&1; echo "BUILD_EXIT=$?" >> "$OUT/STATUS.log"
python3 -c "import deep_gemm" >> "$OUT/build.log" 2>&1 || { echo "IMPORT_FAILED" >> "$OUT/STATUS.log"; exit 1; }
FE78="timeout 7200 python3 tests/test_frontend_fe78.py --seeds $SEEDS --rows 1 2 8 16"
( CUDA_VISIBLE_DEVICES=0 $FE78 --mma auto > "$OUT/eq_auto_defaults.log" 2>&1; echo "GPU0_EXIT=$?" >> "$OUT/STATUS.log" ) &
( CUDA_VISIBLE_DEVICES=1 $FE78 --grid auto --mma cc > "$OUT/eq_auto_cc.log" 2>&1; echo "GPU1_EXIT=$?" >> "$OUT/STATUS.log" ) &
( CUDA_VISIBLE_DEVICES=2 $FE78 --grid 96 --mma swapab --wlayout fragment > "$OUT/eq_96_swapab_fragment.log" 2>&1; echo "GPU2_EXIT=$?" >> "$OUT/STATUS.log" ) &
( export CUDA_VISIBLE_DEVICES=3
  DG_FE_TINYM_GRID=96 DG_FE_TINYM_MMA=wmma timeout 3600 python3 tests/test_frontend_tinym.py --seeds 200 > "$OUT/bitid_96_wmma.log" 2>&1; echo "GPU3a_EXIT=$?" >> "$OUT/STATUS.log"
  timeout 3600 python3 tests/test_frontend_tinym.py --seeds 200 > "$OUT/bitid_defaults.log" 2>&1; echo "GPU3b_EXIT=$?" >> "$OUT/STATUS.log"
  timeout 3600 python3 tests/test_frontend_tinym.py --seeds 200 --l2-persist 1 > "$OUT/bitid_defaults_persist1.log" 2>&1; echo "GPU3c_EXIT=$?" >> "$OUT/STATUS.log" ) &
wait
for f in "$OUT"/eq_*.log "$OUT"/bitid_*.log; do echo "## $(basename "$f")"; grep -E "library defaults|new-scheme|full-K vs legacy|bit-identity|^PASS|^FAIL|Error" "$f" | tail -4; done >> "$OUT/SUMMARY.log"
echo "ALL_DONE $(date)" >> "$OUT/STATUS.log"
