#!/usr/bin/env bash
# DG_FP4_PUSH_DONE_FLAGS A/B (push DONE count replaces NVLink barrier #1), inside four_api_build:
#   MODE=bringup : mxfp4 T=2 knob 1 under DG_FP4_SPIN_TIMEOUT=1 (a protocol bug traps, never hangs)
#   MODE=corr    : tests/test_four_api_correctness.py mxfp4+qoq T=2 8 8 16 twice (knob 1), mxfp4 128/512
#   MODE=stress  : 200-iter graph-replay probe at mxfp4 M=8 (knob 1), then corr T=2 8 8 16 again
#   MODE=perf    : per (quant, M) point, knob 1 then 0, PASSES passes, each cell = phase-stamp
#                  probe under nsys (skew-free min-over-devices kernel duration via
#                  scripts/reconcile_nsys_devices.py + slots 9/1/2/3/16); OFFICIAL=1 adds the
#                  official-style capture on pass 1.
# Waits for idle GPUs (no compute apps, no live nvcc, no other /raid/kimi/results/*_RUNNING marker)
# before every GPU run and holds /raid/kimi/results/pdf_RUNNING while running.
# Usage: MODE=bringup|corr|stress|perf PASSES=2 OFFICIAL=0 OUT=/raid/kimi/results/pdf bash scripts/run_pdf_ab.sh
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${MODE:-corr}; PASSES=${PASSES:-2}; OFFICIAL=${OFFICIAL:-0}
OUT=${OUT:-/raid/kimi/results/pdf}
POINTS=${POINTS:-"mxfp4:2 mxfp4:8 mxfp4:16 qoq:8"}
# perf cells: tag:ENV=V[,ENV=V...]
CONFIGS=${CONFIGS:-"k1:DG_FP4_PUSH_DONE_FLAGS=1 k0:DG_FP4_PUSH_DONE_FLAGS=0"}
# Knob(s) applied to the bringup / corr / stress runs (reused for later knob A/Bs, e.g.
# KNOBS="DG_FP4_PUSH_DET_SLOTS=1" OUT=/raid/kimi/results/slots MARK=.../CAPTURE_SLOTS_RUNNING)
KNOBS=${KNOBS:-DG_FP4_PUSH_DONE_FLAGS=1}
MARK=${MARK:-/raid/kimi/results/pdf_RUNNING}
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/usr/local/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export TORCH_CUDA_ARCH_LIST=9.0a
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_JIT_CACHE_DIR="$ROOT/.jit_cache"
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
mkdir -p "$OUT" /raid/kimi/results
LOG="$OUT/pdf_${MODE}.log"
TR=(/usr/local/bin/torchrun --standalone --nproc_per_node=8)
NSYS=(/usr/local/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none
      --cpuctxsw=none --force-overwrite=true)

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

echo "=== MODE=$MODE build $(git rev-parse --short HEAD) $(date) clocks: $(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits | tr '\n' ' ')" >> "$LOG"

run_corr() {  # api T env...
  local api=$1 T=$2; shift 2
  wait_idle
  echo "--- CORR $api T=$T ($*)" >> "$LOG"
  env "$@" timeout 600 "${TR[@]}" tests/test_four_api_correctness.py --apis $api --tokens "$T" > "$OUT/corr_${MODE}_T${T}.log" 2>&1
  local rc=$?
  grep -h "cos_min\|Error\|error\|Traceback\|timeout\|trap" "$OUT/corr_${MODE}_T${T}.log" | grep -v "^$" | sort -u | head -20 >> "$LOG"
  echo "EXIT=$rc" >> "$LOG"
}

if [ "$MODE" = bringup ]; then
  run_corr mxfp4_mega_moe_fused 2 $KNOBS DG_FP4_SPIN_TIMEOUT=1
  run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" 16 $KNOBS DG_FP4_SPIN_TIMEOUT=1
fi

if [ "$MODE" = corr ]; then
  for rep in 1 2; do
    for T in 2 8 8 16; do
      run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" $T $KNOBS
    done
  done
  run_corr mxfp4_mega_moe_fused 128 $KNOBS
  run_corr mxfp4_mega_moe_fused 512 $KNOBS
fi

if [ "$MODE" = stress ]; then
  wait_idle
  for SM in ${STRESS_MS:-8}; do
    wait_idle
    echo "--- STRESS mxfp4 M=$SM iters=200 ($KNOBS)" >> "$LOG"
    env $KNOBS DG_FP4_SPIN_TIMEOUT=${STRESS_SPIN_TIMEOUT:-0} timeout 900 "${TR[@]}" tests/profile_fused_phase_stamps.py \
      --quant mxfp4 --global-tokens $SM --iters 200 > "$OUT/stress_m$SM.log" 2>&1
    echo "STRESS_EXIT=$?" >> "$LOG"
    sed -n '/=== fused/,$p' "$OUT/stress_m$SM.log" | grep -v "PROBE_ITER\|Warning\|warn" | head -30 >> "$LOG"
  done
  for T in 2 8 8 16; do
    run_corr "mxfp4_mega_moe_fused qoq_mega_moe_fused" $T $KNOBS
  done
fi

if [ "$MODE" = perf ]; then
  for pass in $(seq 1 "$PASSES"); do
    for pt in $POINTS; do
      Q=${pt%%:*}; M=${pt##*:}
      for cfg in $CONFIGS; do
        K=${cfg%%:*}; envs=${cfg#*:}; envs=${envs//,/ }
        name="${Q}_M${M}_${K}_p${pass}"
        wait_idle
        echo "--- PROBE+NSYS $name ($envs)" >> "$LOG"
        env $envs PROBE_DUMP=1 timeout 600 "${NSYS[@]}" --output="$OUT/probe_nsys_$name" "${TR[@]}" \
          tests/profile_fused_phase_stamps.py --quant "$Q" --global-tokens "$M" --iters 20 > "$OUT/probe_nsys_$name.log" 2>&1
        echo "EXIT=$?" >> "$LOG"
        sed -n '/=== fused/,$p' "$OUT/probe_nsys_$name.log" | grep -v "PROBE_ITER\|Warning\|warn" >> "$LOG"
        python3 scripts/reconcile_nsys_devices.py "$OUT/probe_nsys_$name.nsys-rep" 2>&1 | grep -v "^$" >> "$LOG"
        if [ "$OFFICIAL" = 1 ] && [ "$pass" = 1 ]; then
          wait_idle
          echo "--- OFFICIAL $name" >> "$LOG"
          env $envs timeout 600 "${NSYS[@]}" --output="$OUT/official_$name" "${TR[@]}" tests/profile_four_api_h20.py \
            --scope mega --backend fused --quant "$Q" --global-tokens "$M" > "$OUT/official_$name.log" 2>&1
          echo "EXIT=$?" >> "$LOG"
          python3 scripts/reconcile_nsys_devices.py "$OUT/official_$name.nsys-rep" 2>&1 | grep -v "^$" >> "$LOG"
        fi
      done
    done
  done
fi
rm -f "$MARK"
echo "PDF_${MODE}_DONE" >> "$LOG"
