#!/usr/bin/env bash
# Register-budget matrix for the third-math-WG study (docs/third_math_wg.md).
#   regbudget_matrix.sh <outdir> <cores> <name>=<kernel.cu> [<name>=<kernel.cu> ...]
# Runs every (variant x config) compile pinned to <cores> (taskset list), 8 at a time,
# and appends the summary lines to <outdir>/summary.txt.
set -u
out=$1; cores=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
: > "$out/summary.txt"
configs=(
  "w2_e208:"
  "w2_e200:-DDG_FP4_EPI_REGS=200"
  "w2_e192:-DDG_FP4_EPI_REGS=192"
  "w2_e184:-DDG_FP4_EPI_REGS=184"
  "w2_e176:-DDG_FP4_EPI_REGS=176"
  "w2_e168:-DDG_FP4_EPI_REGS=168"
  "w2_e160:-DDG_FP4_EPI_REGS=160"
  "w2_e152:-DDG_FP4_EPI_REGS=152"
  "w2_e144:-DDG_FP4_EPI_REGS=144"
  "w2_e136:-DDG_FP4_EPI_REGS=136"
  "w2_e128:-DDG_FP4_EPI_REGS=128"
  "w2_n40:-DDG_FP4_NONEPI_REGS=40"
  "w2_n32_d32:-DDG_FP4_NONEPI_REGS=32 -DDG_FP4_DISPATCH_REGS=32"
  "w3_e144:-DDG_FP4_MATH_WGS=3"
  "w3_e152_n40:-DDG_FP4_MATH_WGS=3 -DDG_FP4_EPI_REGS=152 -DDG_FP4_NONEPI_REGS=40"
  "w3_e128:-DDG_FP4_MATH_WGS=3 -DDG_FP4_EPI_REGS=128"
  "w3_e160_n24_d24:-DDG_FP4_MATH_WGS=3 -DDG_FP4_EPI_REGS=160 -DDG_FP4_NONEPI_REGS=24 -DDG_FP4_DISPATCH_REGS=24"
)
n=0
for kv in "$@"; do
  name=${kv%%=*}; src=${kv#*=}
  for c in "${configs[@]}"; do
    tag="${name}_${c%%:*}"; flags=${c#*:}
    ( taskset -c "$cores" bash "$here/regbudget_compile.sh" "$src" "$out" "$tag" $flags >> "$out/summary.txt" ) &
    n=$((n+1)); if [ $((n % 8)) -eq 0 ]; then wait; fi
  done
done
wait
echo DONE >> "$out/summary.txt"
