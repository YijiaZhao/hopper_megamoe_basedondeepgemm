#!/usr/bin/env bash
# swapab / fragment-layout FE experiment on an idle 8-GPU H20 (inside the build container, repo root):
#   phase 1: standalone stamps + CUDA-event FE, {grid 96: wmma | swapab | swapab+fragment; grid auto: wmma | swapab}
#            x rows {1, 2} x {mxfp4, qoq} = 20 cells, one per GPU in GPUS, 4 rounds;
#   phase 2: NCU request / sector counts of the router kernel (grid 96, rows 1, mxfp4) for the 3 MMA paths;
#   phase 3: top-8 equality, 1000 seeds, legacy 96 wmma vs {96 swapab, 96 swapab+fragment, auto swapab} (3 GPUs);
#   phase 4: 8-rank test_four_api_correctness once.
# Usage: [GPUS="0 1 2 4 5 6 7"] [OUT=/raid/kimi/results/fe/feswap] bash scripts/run_feswap_matrix.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
OUT=${OUT:-/raid/kimi/results/fe/feswap}; mkdir -p "$OUT"
GPUS=${GPUS:-"0 1 2 4 5 6 7"}; read -r -a G <<< "$GPUS"
ITERS=${ITERS:-100}; SEEDS=${SEEDS:-1000}
export PYTHONPATH="$ROOT" CUDA_HOME=/usr/local/cuda PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST=9.0a
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-/raid/kimi/dg_dev/third-party/cutlass/include}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
echo "=== feswap matrix $(git rev-parse --short HEAD) $(date)" > "$OUT/STATUS.log"
# phase 1
CELLS=()
for R in 1 2; do for Q in mxfp4 qoq; do
  CELLS+=("96 wmma row $R $Q" "96 swapab row $R $Q" "96 swapab fragment $R $Q" "auto wmma row $R $Q" "auto swapab row $R $Q")
done; done
i=0; pids=()
for cell in "${CELLS[@]}"; do
  read -r GR MM WL R Q <<< "$cell"; gpu=${G[$((i % ${#G[@]}))]}
  if [ $i -ge ${#G[@]} ]; then wait "${pids[$((i - ${#G[@]}))]}"; fi
  CUDA_VISIBLE_DEVICES=$gpu DG_FE_TINYM_GRID=$GR DG_FE_TINYM_MMA=$MM DG_FE_ROUTER_WLAYOUT=$WL timeout 900 \
    python3 tests/fe_standalone_bench.py --quant "$Q" --rows "$R" --iters "$ITERS" > "$OUT/g${GR}_${MM}_${WL}_r${R}_${Q}.log" 2>&1 &
  pids+=($!); i=$((i + 1))
done
wait
{ for cell in "${CELLS[@]}"; do read -r GR MM WL R Q <<< "$cell"; echo "##### grid=$GR mma=$MM wlayout=$WL rows=$R quant=$Q"; cat "$OUT/g${GR}_${MM}_${WL}_r${R}_${Q}.log"; echo; done; } > "$OUT/MATRIX.log"
echo "PHASE1_DONE $(date)" >> "$OUT/STATUS.log"
# phase 2: NCU (GPU 7 = last of GPUS)
NG=${G[$((${#G[@]} - 1))]}
MET="gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ldgsts.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts.sum,l1tex__t_requests.sum,l1tex__t_sectors.sum,lts__t_requests_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_requests.sum,lts__t_sectors.sum,dram__sectors_read.sum,dram__bytes_read.sum,smsp__inst_executed_op_global_ld.sum,sm__inst_executed_pipe_lsu.sum"
for cfg in "wmma row" "swapab row" "swapab fragment"; do
  read -r MM WL <<< "$cfg"
  CUDA_VISIBLE_DEVICES=$NG DG_FE_TINYM_GRID=96 DG_FE_TINYM_MMA=$MM DG_FE_ROUTER_WLAYOUT=$WL timeout 900 \
    ncu --kernel-name regex:router_quant_topk_kernel --launch-skip 5 --launch-count 1 --clock-control none --metrics "$MET" \
    python3 tests/ncu_frontend_tinym.py --quant mxfp4 --rows 1 --mma "$MM" > "$OUT/ncu_96_${MM}_${WL}_r1_mxfp4.log" 2>&1
done
echo "PHASE2_DONE $(date)" >> "$OUT/STATUS.log"
# phase 3: equality, 3 GPUs in parallel
CUDA_VISIBLE_DEVICES=${G[0]} timeout 3600 python3 tests/test_frontend_fe78.py --seeds "$SEEDS" --grid 96 --mma swapab > "$OUT/eq_96_swapab_row.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=${G[1]} timeout 3600 python3 tests/test_frontend_fe78.py --seeds "$SEEDS" --grid 96 --mma swapab --wlayout fragment > "$OUT/eq_96_swapab_fragment.log" 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=${G[2]} timeout 3600 python3 tests/test_frontend_fe78.py --seeds "$SEEDS" --grid auto --mma swapab > "$OUT/eq_auto_swapab_row.log" 2>&1 &
p3=$!
wait $p1 $p2 $p3
echo "PHASE3_DONE $(date)" >> "$OUT/STATUS.log"
# phase 4: 8-rank correctness once (all 8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DG_FE_TINYM_MMA=swapab timeout 900 /usr/local/bin/torchrun --standalone --nproc_per_node=8 \
  tests/test_four_api_correctness.py --tokens 8 > "$OUT/corr8_tokens8.log" 2>&1
echo "corr EXIT=$?" >> "$OUT/STATUS.log"
echo "ALL_DONE $(date)" >> "$OUT/STATUS.log"
rm -f /raid/kimi/results/CAPTURE_FESWAP_RUNNING
