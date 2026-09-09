#!/usr/bin/env bash
# L2-tail probe A/B: phase-stamp probe (rank 0, slots 3/4/36/37/40/5/6/7) for a list of
# (quant, M) with the knob assignments in KNOBS_OFF / KNOBS_ON (space separated VAR=VAL),
# PASSES passes (default 2, second pass in reverse order). Honours the RUNNING markers.
# Usage (inside four_api_build): KNOBS_ON="DG_FP4_SPLITK_L2=1" bash scripts/run_l2tail_probe.sh OUTDIR mxfp4:8 mxfp4:2 mxfp4:16 qoq:8
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
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW
RES=/raid/kimi/results
PASSES=${PASSES:-2}
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
one() {  # $1 tag $2 quant $3 M $4 pass, rest = env
  local tag=$1 quant=$2 m=$3 pass=$4; shift 4
  local log="$OUT/probe_${quant}_m${m}_${tag}_p${pass}.log"
  wait_idle || return 1
  env "$@" timeout 600 /usr/local/bin/torchrun --standalone --nproc_per_node=8 \
    tests/profile_fused_phase_stamps.py --quant "$quant" --global-tokens "$m" --iters 20 > "$log" 2>&1
  echo "EXIT=$?" >> "$log"
  printf '%-6s %-5s M%-3s p%s | ' "$tag" "$quant" "$m" "$pass"
  awk '/^ *(3|4|36|37|40|5|6|7) /{printf "%s=%s ", $1, $(NF-3)} /CUDA-event wall/{printf "wall=%s", $5} END{print ""}' "$log"
}
echo "build $(git rev-parse --short HEAD) clocks $(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits | tr '\n' ' ')"
echo "OFF: ${KNOBS_OFF:-<default>}   ON: ${KNOBS_ON:-<none>}"
for pass in $(seq 1 "$PASSES"); do
  items=("$@"); [ $((pass % 2)) -eq 0 ] && items=($(printf '%s\n' "$@" | tac))
  for it in "${items[@]}"; do
    q=${it%%:*}; m=${it##*:}
    one off "$q" "$m" "$pass" ${KNOBS_OFF:-}
    [ -n "${KNOBS_ON:-}" ] && one on "$q" "$m" "$pass" ${KNOBS_ON}
  done
done
echo L2TAIL_PROBE_DONE
