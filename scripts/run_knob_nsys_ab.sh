#!/usr/bin/env bash
# Skew-free official-method knob A/B: nsys (official flags) around profile_four_api_h20.py
# --scope mega --backend fused with DG_PROFILE_HOST_BARRIER=1, knob OFF vs ON, PASSES passes
# (pass 2 reversed), then reconcile_nsys_devices.py (GPU0 / min-over-dev / skew per report).
# Usage (inside four_api_build): KNOB=DG_FP4_SPLITK_L2 OFF=0 ON=3 bash scripts/run_knob_nsys_ab.sh OUTDIR mxfp4:8 mxfp4:2 mxfp4:16 qoq:8
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
OUT=$1; shift
export PYTHONPATH="$ROOT"
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/local/cuda
export DG_CUTLASS_INCLUDE_PATH="$ROOT/third-party/cutlass/include"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_BENCH_FLUSH_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES:-268435456}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_four_api}
export DG_PROFILE_HOST_BARRIER=${DG_PROFILE_HOST_BARRIER:-1}
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
RES=/raid/kimi/results
PASSES=${PASSES:-2}
KNOB=${KNOB:?}; OFF=${OFF:-0}; ON=${ON:-1}
mkdir -p "$OUT"
wait_idle() {
  local i
  for i in $(seq 1 300); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] &&
       ! ls "$RES"/OFFICIAL_*_RUNNING "$RES"/CAPTURE_RUNNING >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "GPUs busy after 600 s" >&2; return 1
}
NSYS=(/usr/local/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none
      --cpuctxsw=none --force-overwrite=true)
echo "build $(git rev-parse --short HEAD) clocks $(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits | tr '\n' ' ') host barrier $DG_PROFILE_HOST_BARRIER"
echo "$$ $(date)" > "$RES/CAPTURE_RUNNING"
trap 'rm -f "$RES/CAPTURE_RUNNING"' EXIT
for pass in $(seq 1 "$PASSES"); do
  items=("$@"); [ $((pass % 2)) -eq 0 ] && items=($(printf '%s\n' "$@" | tac))
  for it in "${items[@]}"; do
    q=${it%%:*}; m=${it##*:}
    for side in off on; do
      v=$OFF; [ "$side" = on ] && v=$ON
      name="${q}_M${m}_${side}_p${pass}"
      wait_idle || exit 1
      env "$KNOB=$v" timeout 600 "${NSYS[@]}" --output="$OUT/$name" \
        /usr/local/bin/torchrun --standalone --nproc_per_node=8 tests/profile_four_api_h20.py \
        --scope mega --backend fused --quant "$q" --global-tokens "$m" > "$OUT/$name.log" 2>&1
      echo "$name EXIT=$?"
    done
  done
done
python3 scripts/reconcile_nsys_devices.py --last 8 "$OUT"/*.nsys-rep > "$OUT/reconcile.txt" 2>&1
grep -E "^##|last3" "$OUT/reconcile.txt"
echo KNOB_NSYS_AB_DONE
