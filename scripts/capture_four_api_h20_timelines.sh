#!/usr/bin/env bash
# Capture the complete H20 delivery matrix (2 scopes x 2 backends x 2 quants x
# TOKENS_LIST). The customer method is TOKENS_LIST="2 8 16" -> 24 reports.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
OUT=${OUT:-$ROOT/artifacts/four_api_h20_nsys}
FORCE=${FORCE:-0}
GPU_IDLE_LIMIT_MIB=${GPU_IDLE_LIMIT_MIB:-64}
GPU_IDLE_RETRIES=${GPU_IDLE_RETRIES:-45}
LOCK_SM_CLOCK_MHZ=${LOCK_SM_CLOCK_MHZ:-1830}
CLOCK_LOCK_MODE=${CLOCK_LOCK_MODE:-verify}
TOKENS_LIST=${TOKENS_LIST:-"2 8 16"}
read -r -a TOKENS <<< "$TOKENS_LIST"
# Optional sub-matrix (default = full customer matrix): SCOPES="e2e" BACKENDS="fused"
SCOPES=${SCOPES:-"e2e mega"}
BACKENDS=${BACKENDS:-"split fused"}
QUANTS=${QUANTS:-"mxfp4 qoq"}
read -r -a _SC <<< "$SCOPES"; read -r -a _BK <<< "$BACKENDS"; read -r -a _QU <<< "$QUANTS"
EXPECTED_COUNT=$((${#_SC[@]} * ${#_BK[@]} * ${#_QU[@]} * ${#TOKENS[@]}))
export EXPECTED_PER_M=$((${#_SC[@]} * ${#_BK[@]} * ${#_QU[@]}))

cd "$ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export DG_CUTLASS_INCLUDE_PATH=${DG_CUTLASS_INCLUDE_PATH:-$ROOT/third-party/cutlass/include}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW

for command in nvidia-smi nsys torchrun python3; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done

check_idle_gpus() {
  local attempt i
  local -a used
  for attempt in $(seq 1 "$GPU_IDLE_RETRIES"); do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "${#used[@]}" -ne 8 ]; then
      echo "expected 8 visible GPUs, found ${#used[@]}" >&2
      return 1
    fi
    for i in "${!used[@]}"; do
      if [ "${used[$i]}" -gt "$GPU_IDLE_LIMIT_MIB" ]; then break; fi
    done
    if [ "$i" -eq 7 ] && [ "${used[$i]}" -le "$GPU_IDLE_LIMIT_MIB" ]; then
      return 0
    fi
    if nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
         --format=csv,noheader,nounits | grep -q .; then
      echo "another GPU process is active" >&2
      nvidia-smi >&2
      return 1
    fi
    sleep 1
  done
  echo "GPU memory did not return to idle" >&2
  nvidia-smi >&2
  return 1
}

# RESUME=1: keep the existing reports and capture only the missing cases (a case whose
# post-capture idle check fails is deleted, so kept reports are always clean).
RESUME=${RESUME:-0}
if [ "$RESUME" != 1 ] && find "$OUT" -maxdepth 1 -name '*.nsys-rep' -print -quit 2>/dev/null | grep -q .; then
  if [ "$FORCE" != 1 ]; then
    echo "output contains existing .nsys-rep files: $OUT (set FORCE=1 to replace)" >&2
    exit 3
  fi
  python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
for path in Path(sys.argv[1]).glob('*.nsys-rep'):
    path.unlink()
for name in ('SHA256SUMS', 'VERIFY.json', 'TIMELINE_TABLE.csv',
             'TIMELINE_TABLE.json', 'TIMELINE_TABLE.md'):
    (Path(sys.argv[1]) / name).unlink(missing_ok=True)
PY
fi
mkdir -p "$OUT"
check_idle_gpus
case "$CLOCK_LOCK_MODE" in
  set) nvidia-smi -lgc "$LOCK_SM_CLOCK_MHZ,$LOCK_SM_CLOCK_MHZ" ;;
  verify) ;;
  *) echo "CLOCK_LOCK_MODE must be set or verify" >&2; exit 2 ;;
esac
mapfile -t locked_clocks < <(
  nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits
)
if [ "${#locked_clocks[@]}" -ne 8 ]; then
  echo "expected eight clock readings" >&2
  exit 2
fi
for clock in "${locked_clocks[@]}"; do
  if [ "$clock" -ne "$LOCK_SM_CLOCK_MHZ" ]; then
    echo "GPU clock is not locked at $LOCK_SM_CLOCK_MHZ MHz; lock it on the host first" >&2
    exit 3
  fi
done
nvidia-smi dmon -s c -d 1 > "$OUT/CLOCK_DMON.txt" 2>&1 &
CLOCK_MONITOR_PID=$!
stop_clock_monitor() {
  kill "$CLOCK_MONITOR_PID" 2>/dev/null || true
  wait "$CLOCK_MONITOR_PID" 2>/dev/null || true
}
trap stop_clock_monitor EXIT

COMMON=(--trace=cuda,nvtx --cuda-graph-trace=node --sample=none
        --cpuctxsw=none --force-overwrite=true)
run_case() {
  local name=$1 scope=$2 backend=$3 quant=$4 tokens=$5
  if [ "$RESUME" = 1 ] && [ -f "$OUT/$name.nsys-rep" ]; then
    echo "=== $name === (kept)"
    return 0
  fi
  check_idle_gpus
  echo "=== $name ==="
  nsys profile "${COMMON[@]}" --output="$OUT/$name" \
    torchrun --standalone --nproc_per_node=8 tests/profile_four_api_h20.py \
      --scope "$scope" --backend "$backend" --quant "$quant" \
      --global-tokens "$tokens"
  # Another job appearing mid-case contaminates it: drop the report before failing
  if ! check_idle_gpus; then
    rm -f "$OUT/$name.nsys-rep"
    return 1
  fi
}

for scope in $SCOPES; do
  for backend in $BACKENDS; do
    for quant in $QUANTS; do
      for tokens in "${TOKENS[@]}"; do
        run_case "${scope}_${backend}_${quant}_M${tokens}" \
          "$scope" "$backend" "$quant" "$tokens"
      done
    done
  done
done

stop_clock_monitor
trap - EXIT
awk -v expected="$LOCK_SM_CLOCK_MHZ" '!/^#/ && NF {n++; if ($3 != expected) bad++} END {exit(bad != 0 || n == 0)}' "$OUT/CLOCK_DMON.txt"

find "$OUT" -maxdepth 1 -type f -name '*.nsys-rep' -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
python3 scripts/verify_four_api_h20_timelines.py "$OUT"
python3 scripts/summarize_four_api_h20_timelines.py "$OUT" | tee "$OUT/TIMELINE_TABLE.md"
python3 scripts/summarize_four_api_h20_last3.py "$OUT" >/dev/null
count=$(find "$OUT" -maxdepth 1 -type f -name '*.nsys-rep' | wc -l)
echo "TIMELINE_COUNT=$count"
test "$count" -eq "$EXPECTED_COUNT"
