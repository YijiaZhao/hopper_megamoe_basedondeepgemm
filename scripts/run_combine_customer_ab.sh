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
echo "$$ $(date)" > "$RES/CAPTURE_MCMB_RUNNING"
trap 'rm -f "$RES/CAPTURE_MCMB_RUNNING"' EXIT
wait_idle() {
  local i
  for i in $(seq 1 1800); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] &&
       ! ls "$RES"/OFFICIAL_*_RUNNING "$RES"/CAPTURE_RUNNING "$RES"/CAPTURE_FUSE_RUNNING >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "GPUs busy after 1 h" >&2; return 1
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
    wait_idle || exit 1
    echo "=== customer capture knob=$knob pass=$pass -> $OUT $(date)"
    OUT="$OUT" FORCE=1 DG_FP4_COMBINE_DYNAMIC=$knob timeout 3600 bash scripts/capture_four_api_h20_timelines.sh > "$OUT.log" 2>&1
    echo "CAPTURE_EXIT=$?"
    for attempt in $(seq 1 "$RECAPTURE_MAX"); do
      bad=$(skewed_points "$OUT")
      [ -z "$bad" ] && break
      echo "--- recapture (attempt $attempt): $(echo "$bad" | tr '\n' ';')"
      while read -r name skew; do
        [ -z "$name" ] && continue
        q=$(echo "$name" | cut -d_ -f3); m=$(echo "$name" | sed -E 's/.*_M([0-9]+)$/\1/')
        wait_idle || exit 1
        DG_FP4_COMBINE_DYNAMIC=$knob timeout 600 "${NSYS[@]}" --output="$OUT/$name" \
          /usr/local/bin/torchrun --standalone --nproc_per_node=8 tests/profile_four_api_h20.py \
          --scope mega --backend fused --quant "$q" --global-tokens "$m" >> "$OUT.log" 2>&1
        echo "recapture $name EXIT=$?"
      done <<< "$bad"
    done
    skewed_points "$OUT" | sed 's/^/STILL_SKEWED /'
    grep -E "^##|last3" "$OUT/reconcile_fused.txt" | paste - - | sed -E 's/.nsys-rep: kernels=.*launches.device=.[0-9, ]*.//; s/last3 .official window.//'
    python3 scripts/summarize_four_api_h20_last3.py "$OUT" > /dev/null 2>&1
    grep -E "^scope|^mega,.*,fused" "$OUT/TIMELINE_LAST3.csv" 2>/dev/null
  done
done
echo CUSTOMER_AB_DONE
