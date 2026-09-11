#!/usr/bin/env bash
# Single-GPU relative A/B of the swapab / fragment-layout FE variants on a ComputeLab H20 (HBM3,
# not the H20-3e of .7 -- compare only within this node). Runs inside a torch+nvcc container
# (lmsysorg/sglang:dev): clone -> build -> stamps/event matrix (sequential, one GPU) -> equality.
#   docker run -d --gpus device=0 --ipc=host -v $HOME/feswap_out:/out lmsysorg/sglang:dev \
#     bash -c "git clone -b perf/fe-78cta https://github.com/YijiaZhao/hopper_megamoe_basedondeepgemm.git /work/repo && bash /work/repo/scripts/run_feswap_computelab.sh"
set -uo pipefail
OUT=${OUT:-/out}; mkdir -p "$OUT"; ROOT=${ROOT:-/work/repo}; cd "$ROOT" || exit 1
SEEDS=${SEEDS:-300}; ITERS=${ITERS:-100}
export PYTHONPATH="$ROOT" CUDA_HOME=/usr/local/cuda PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST=9.0a
export PATH="/usr/local/cuda/bin:$PATH" DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
echo "=== computelab feswap $(git rev-parse --short HEAD) $(date)" > "$OUT/STATUS.log"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.mem --format=csv >> "$OUT/STATUS.log"
git submodule update --init --depth 1 third-party/cutlass third-party/fmt >> "$OUT/build.log" 2>&1
bash develop.sh >> "$OUT/build.log" 2>&1; echo "BUILD_EXIT=$?" >> "$OUT/STATUS.log"
python3 -c "import deep_gemm" >> "$OUT/build.log" 2>&1 || { echo "IMPORT_FAILED" >> "$OUT/STATUS.log"; exit 1; }
CELLS=()
for R in 1 2; do for Q in mxfp4 qoq; do
  CELLS+=("96 wmma row $R $Q" "96 swapab row $R $Q" "96 swapab fragment $R $Q" "auto wmma row $R $Q" "auto swapab row $R $Q")
done; done
for cell in "${CELLS[@]}"; do
  read -r GR MM WL R Q <<< "$cell"
  echo "##### grid=$GR mma=$MM wlayout=$WL rows=$R quant=$Q" >> "$OUT/MATRIX.log"
  DG_FE_TINYM_GRID=$GR DG_FE_TINYM_MMA=$MM DG_FE_ROUTER_WLAYOUT=$WL timeout 900 \
    python3 tests/fe_standalone_bench.py --quant "$Q" --rows "$R" --iters "$ITERS" >> "$OUT/MATRIX.log" 2>&1
  echo >> "$OUT/MATRIX.log"
done
echo "PHASE1_DONE $(date)" >> "$OUT/STATUS.log"
timeout 3600 python3 tests/test_frontend_fe78.py --seeds "$SEEDS" --grid 96 --mma swapab > "$OUT/eq_96_swapab_row.log" 2>&1
timeout 3600 python3 tests/test_frontend_fe78.py --seeds "$SEEDS" --grid 96 --mma swapab --wlayout fragment > "$OUT/eq_96_swapab_fragment.log" 2>&1
timeout 3600 python3 tests/test_frontend_fe78.py --seeds "$SEEDS" --grid auto --mma swapab > "$OUT/eq_auto_swapab_row.log" 2>&1
echo "ALL_DONE $(date)" >> "$OUT/STATUS.log"
