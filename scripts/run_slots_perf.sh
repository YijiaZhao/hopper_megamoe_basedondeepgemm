#!/usr/bin/env bash
# DG_FP4_PUSH_DET_SLOTS A/B, CUDA-event method (tests/bench_frontend_tinym.py, Mega column:
# event0 -> Mega graph replay -> event1 after an L2 flush + all-rank host barrier, GPU0
# median/min/p90 over ITERS), knob 1 then 0 per (quant, M) point, PASSES passes.
# Usage (inside four_api_build): PASSES=2 ITERS=100 OUT=/raid/kimi/results/slots bash scripts/run_slots_perf.sh
# Waits for idle GPUs / no foreign *_RUNNING marker; holds $MARK while running.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PASSES=${PASSES:-2}; ITERS=${ITERS:-100}
OUT=${OUT:-/raid/kimi/results/slots}
POINTS=${POINTS:-"mxfp4:2 mxfp4:8 mxfp4:16 qoq:2 qoq:8 qoq:16"}
CONFIGS=${CONFIGS:-"k1:DG_FP4_PUSH_DET_SLOTS=1 k0:DG_FP4_PUSH_DET_SLOTS=0"}
MARK=${MARK:-/raid/kimi/results/CAPTURE_SLOTS_RUNNING}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH="${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}"
export TORCH_CUDA_ARCH_LIST=9.0a
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
mkdir -p "$OUT" /raid/kimi/results
LOG="$OUT/slots_perf_events.log"
TR=(/usr/local/bin/torchrun --standalone --nproc_per_node=8)
wait_idle() {
  local i
  rm -f "$MARK"
  for i in $(seq 1 900); do
    local apps nvcc others
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    nvcc=$(ps -C nvcc -o stat= 2>/dev/null | grep -v '^Z' | grep -c .)
    others=$(ls /raid/kimi/results/*_RUNNING 2>/dev/null | grep -v "$(basename "$MARK")" | grep -c .)
    if [ "$apps" = 0 ] && [ "$nvcc" = 0 ] && [ "$others" = 0 ]; then touch "$MARK"; return 0; fi
    sleep 2
  done
  echo "GPUs still busy after 1800 s" >&2; return 1
}
trap 'rm -f "$MARK"' EXIT
echo "=== events build $(git rev-parse --short HEAD) $(date) clocks: $(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits | tr '\n' ' ')" >> "$LOG"
for pass in $(seq 1 "$PASSES"); do
  for pt in $POINTS; do
    Q=${pt%%:*}; M=${pt##*:}
    for cfg in $CONFIGS; do
      K=${cfg%%:*}; envs=${cfg#*:}; envs=${envs//,/ }
      name="${Q}_M${M}_${K}_p${pass}"
      wait_idle
      env $envs timeout 900 "${TR[@]}" tests/bench_frontend_tinym.py --quant "$Q" --global-tokens "$M" \
        --iters "$ITERS" > "$OUT/events_$name.log" 2>&1
      rc=$?
      echo "--- $name ($envs) EXIT=$rc :: $(grep -m1 '^ *Mega ' "$OUT/events_$name.log" | sed 's/^ *//')" >> "$LOG"
    done
  done
done
rm -f "$MARK"
echo "SLOTS_EVENTS_DONE" >> "$LOG"
