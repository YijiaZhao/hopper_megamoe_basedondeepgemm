#!/usr/bin/env bash
# Offline reproduction of the JIT nvcc line with `-Xptxas -v`, for register-budget
# studies (docs/third_math_wg.md).
#   regbudget_compile.sh <kernel.cu> <outdir> <tag> [extra nvcc flags...]
# <kernel.cu> is a cached JIT translation unit (~/.deep_gemm/cache/kernel.*/kernel.cu);
# the headers are taken from THIS checkout, so header edits are picked up.
# Prints one summary line: tag, rc, registers, spill bytes.
set -u
src=$1; out=$2; tag=$3; shift 3
root=$(cd "$(dirname "$0")/.." && pwd)
cutlass=${DG_CUTLASS_INCLUDE_PATH:-$root/third-party/cutlass/include}
mkdir -p "$out"
nvcc "$src" -cubin -o "$out/$tag.cubin" -std=c++20 --diag-suppress=39,161,174,177,186,940 \
  --ptxas-options=--register-usage-level=10 --ptxas-options=--verbose,--warn-on-local-memory-usage \
  -I"$root/deep_gemm/include" -I"$cutlass" --gpu-architecture=sm_90a \
  --compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi \
  -O3 --expt-relaxed-constexpr --expt-extended-lambda "$@" > "$out/$tag.log" 2>&1
rc=$?
summary=$(grep -E 'spill|Used [0-9]+ registers|error' "$out/$tag.log" | grep -v 'detected during' | tr -s ' ' | tr '\n' ' ')
echo "$tag rc=$rc $summary"
