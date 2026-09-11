#!/usr/bin/env bash
# DG_FP4_L2_PREFETCH_ALL A/B (comm-window L2 weight prefetch), inside four_api_build:
#   MODE=corr : tests/test_four_api_correctness.py mxfp4+qoq T=2 8 8 16 (knob 1), mxfp4 128/512
#   MODE=perf : per (quant, M) point, knob 1 then 0, PASSES passes, each cell =
#               phase-stamp probe under nsys (skew-free min-over-devices kernel duration via
#               scripts/reconcile_nsys_devices.py + stage slots 17/22 + L1 phase) and, if
#               OFFICIAL=1, the official-style nsys capture (profile_four_api_h20.py).
# Usage: MODE=corr|perf|all PASSES=2 OFFICIAL=1 OUT=/raid/kimi/results/l2pf bash scripts/run_l2pf_ab.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-all}; PASSES=${PASSES:-2}; OFFICIAL=${OFFICIAL:-1}
OUT=${OUT:-/raid/kimi/results/l2pf}
POINTS=${POINTS:-"mxfp4:2 mxfp4:8 mxfp4:16 qoq:8 qoq:16"}
# knob configurations: tag:ENV=V,ENV=V ...
CONFIGS=${CONFIGS:-"k1:DG_FP4_L2_PREFETCH_ALL=1 k0:DG_FP4_L2_PREFETCH_ALL=0"}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
mkdir -p "$OUT"
LOG="$OUT/l2pf_ab.log"
TR=(/usr/local/bin/torchrun --standalone --nproc_per_node=8)
NSYS=(/usr/local/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none
      --cpuctxsw=none --force-overwrite=true)

wait_idle() {  # never run on top of someone else's GPU job; give up after WAIT_MAX_S (default 3 h)
  local i
  for i in $(seq 1 $(( ${WAIT_MAX_S:-10800} / 5 ))); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0
    sleep 5
  done
  echo "GPUs still busy after ${WAIT_MAX_S:-10800} s, aborting" >> "$LOG"; echo L2PF_AB_ABORTED >> "$LOG"; exit 1
}

echo "=== build $(git rev-parse --short HEAD) $(date) clocks: $(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits | tr '\n' ' ')" >> "$LOG"

run_corr() {  # api T env...
  local api=$1 T=$2; shift 2
  wait_idle
  echo "--- CORR $api T=$T ($*)" >> "$LOG"
  env "$@" timeout 600 "${TR[@]}" tests/test_four_api_correctness.py --apis $api --tokens "$T" > "$OUT/corr_tmp.log" 2>&1
  local rc=$?
  grep -h "cos_min\|Error\|error\|Traceback" "$OUT/corr_tmp.log" | grep -v "^$" | sort -u | head -20 >> "$LOG"
  echo "EXIT=$rc" >> "$LOG"
}

if [ "$MODE" = corr ] || [ "$MODE" = all ]; then
  for T in 2 8 8 16; do
    run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" $T DG_FP4_L2_PREFETCH_ALL=1
  done
  run_corr mxfp4_mega_moe_fused 128 DG_FP4_L2_PREFETCH_ALL=1
  run_corr mxfp4_mega_moe_fused 512 DG_FP4_L2_PREFETCH_ALL=1
  echo ALL_CORR_DONE >> "$LOG"
fi

if [ "$MODE" = perf ] || [ "$MODE" = all ]; then
  for pass in $(seq 1 "$PASSES"); do
    for pt in $POINTS; do
      Q=${pt%%:*}; M=${pt##*:}
      for CFG in $CONFIGS; do
        K=${CFG%%:*}; ENVS=$(echo "${CFG#*:}" | tr ',' ' ')
        name="${Q}_M${M}_${K}_p${pass}"
        wait_idle
        echo "--- PROBE+NSYS $name ($ENVS)" >> "$LOG"
        env $ENVS PROBE_DUMP=1 timeout 600 "${NSYS[@]}" --output="$OUT/probe_nsys_$name" "${TR[@]}" \
          tests/profile_fused_phase_stamps.py --quant "$Q" --global-tokens "$M" --iters 20 > "$OUT/probe_nsys_$name.log" 2>&1
        echo "EXIT=$?" >> "$LOG"
        sed -n '/=== fused/,$p' "$OUT/probe_nsys_$name.log" | grep -v "PROBE_ITER\|Warning\|warn" >> "$LOG"
        python3 scripts/reconcile_nsys_devices.py "$OUT/probe_nsys_$name.nsys-rep" 2>&1 | grep -v "^$" >> "$LOG"
        if [ "$OFFICIAL" = 1 ] && [ "$pass" = 1 ]; then
          wait_idle
          echo "--- OFFICIAL $name" >> "$LOG"
          env $ENVS timeout 600 "${NSYS[@]}" --output="$OUT/official_$name" "${TR[@]}" tests/profile_four_api_h20.py \
            --scope mega --backend fused --quant "$Q" --global-tokens "$M" > "$OUT/official_$name.log" 2>&1
          echo "EXIT=$?" >> "$LOG"
          python3 scripts/reconcile_nsys_devices.py "$OUT/official_$name.nsys-rep" 2>&1 | grep -v "^$" >> "$LOG"
        fi
      done
    done
  done
  echo ALL_PERF_DONE >> "$LOG"
fi
echo L2PF_AB_DONE >> "$LOG"
