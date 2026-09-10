#!/usr/bin/env bash
# DG_FP4_COMBINE_DYNAMIC perf A/B, customer method first: the official capture
# (scripts/capture_four_api_h20_timelines.sh, full 2 scopes x 2 backends x 2 quants x
# TOKENS_LIST matrix, GPU0 median of the last 3 spans via summarize_four_api_h20_last3.py),
# knob 0 / 1, PASSES passes (pass 2 in reverse knob order), each into its own OUT dir under
# $RES. After each capture, reconcile_nsys_devices.py on the mega fused reports; any point
# whose start skew > SKEW_LIMIT us is re-captured (up to RECAPTURE_MAX times) before the
# last-3 table is (re)built. The skew-free min-over-devices A/B is the footnote
# (run separately: run_combine_dynamic_validate.sh perf).
# Usage (inside four_api_build, GPUs idle, clocks lockable): bash scripts/run_combine_customer_ab.sh [tag]
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TAG=${1:-}
RES=${RES:-/raid/kimi/results}
PASSES=${PASSES:-2}
SKEW_LIMIT=${SKEW_LIMIT:-20}
RECAPTURE_MAX=${RECAPTURE_MAX:-2}
export TOKENS_LIST=${TOKENS_LIST:-"2 8 16"}
export CLOCK_LOCK_MODE=${CLOCK_LOCK_MODE:-set}
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
export PYTHONUNBUFFERED=1
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
mkdir -p "$RES"
MARK="$RES/CAPTURE_MCMB_RUNNING"
trap 'rm -f "$MARK"' EXIT
# Strict exclusivity: before each capture wait until NO other *_RUNNING marker exists and
# the GPUs are empty (60 s of continuous quiet), and only then publish our marker; the
# marker is removed right after the capture. A capture during which another marker
# appeared or another GPU process ran is discarded and re-run (see overlap_watch).
other_markers() { ls "$RES"/*_RUNNING 2>/dev/null | grep -v "CAPTURE_MCMB_RUNNING" || true; }
wait_exclusive() {
  local i quiet=0
  for i in $(seq 1 7200); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ -z "$(other_markers)" ]; then
      quiet=$((quiet + 5)); [ "$quiet" -ge 60 ] && return 0
    else
      quiet=0
    fi
    sleep 5
  done
  echo "GPUs / markers busy after 10 h" >&2; return 1
}
wait_idle() { wait_exclusive; }
# Background watcher during a capture: logs any foreign marker or foreign GPU process
overlap_watch() {  # $1 = overlap file
  : > "$1"
  while true; do
    other_markers >> "$1"
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      case "$(readlink /proc/$p/cwd 2>/dev/null)" in "$ROOT"*) ;; *) echo "pid $p cwd $(readlink /proc/$p/cwd 2>/dev/null)" >> "$1" ;; esac
    done
    sleep 5
  done
}
NSYS=(/usr/local/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none
      --cpuctxsw=none --force-overwrite=true)
# Print "name skew" for every mega fused report whose start skew exceeds the limit
skewed_points() {  # $1 = OUT
  python3 scripts/reconcile_nsys_devices.py --last 3 "$1"/mega_fused_*.nsys-rep > "$1/reconcile_fused.txt" 2>&1
  awk -v lim="$SKEW_LIMIT" '
    /^## / { name = $2; sub(/\.nsys-rep:.*/, "", name) }
    /last3/ { for (i = 1; i <= NF; i++) if ($i == "start" && $(i+1) == "skew") { s = $(i+2) + 0; if (s > lim) print name, s } }
  ' "$1/reconcile_fused.txt"
}
echo "build $(git rev-parse --short HEAD) $(date)"
for pass in $(seq 1 "$PASSES"); do
  knobs="0 1"; [ $((pass % 2)) -eq 0 ] && knobs="1 0"
  for knob in $knobs; do
    OUT="$RES/cmb_customer${TAG}_k${knob}_p${pass}"
    for try in 1 2 3 4; do
      wait_exclusive || exit 1
      echo "$$ $(date)" > "$MARK"
      echo "=== customer capture knob=$knob pass=$pass try=$try -> $OUT $(date)"
      overlap_watch "$OUT.overlap" & watch_pid=$!
      OUT="$OUT" FORCE=1 DG_FP4_COMBINE_DYNAMIC=$knob timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUT.log" 2>&1
      rc=$?
      kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null
      rm -f "$MARK"
      echo "CAPTURE_EXIT=$rc"
      if [ -s "$OUT.overlap" ]; then
        echo "OVERLAP: discarding capture (foreign marker/process during capture): $(sort -u "$OUT.overlap" | tr '\n' ';')"
        rm -rf "$OUT"; continue
      fi
      [ "$rc" = 0 ] && break
    done
    for attempt in $(seq 1 "$RECAPTURE_MAX"); do
      bad=$(skewed_points "$OUT")
      [ -z "$bad" ] && break
      echo "--- recapture (attempt $attempt): $(echo "$bad" | tr '\n' ';')"
      while read -r name skew; do
        [ -z "$name" ] && continue
        q=$(echo "$name" | cut -d_ -f3); m=$(echo "$name" | sed -E 's/.*_M([0-9]+)$/\1/')
        wait_exclusive || exit 1
        echo "$$ $(date)" > "$MARK"
        DG_FP4_COMBINE_DYNAMIC=$knob timeout 600 "${NSYS[@]}" --output="$OUT/$name" \
          /usr/local/bin/torchrun --standalone --nproc_per_node=8 tests/profile_four_api_h20.py \
          --scope mega --backend fused --quant "$q" --global-tokens "$m" >> "$OUT.log" 2>&1
        echo "recapture $name EXIT=$? foreign_markers=[$(other_markers | tr '\n' ' ')]"
        rm -f "$MARK"
      done <<< "$bad"
    done
    skewed_points "$OUT" | sed 's/^/STILL_SKEWED /'
    grep -E "^##|last3" "$OUT/reconcile_fused.txt" | paste - - | sed -E 's/.nsys-rep: kernels=.*launches.device=.[0-9, ]*.//; s/last3 .official window.//'
    python3 scripts/summarize_four_api_h20_last3.py "$OUT" > /dev/null 2>&1
    grep -E "^scope|^mega,.*,fused" "$OUT/TIMELINE_LAST3.csv" 2>/dev/null
  done
done
echo CUSTOMER_AB_DONE
