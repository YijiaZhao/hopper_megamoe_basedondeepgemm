#!/usr/bin/env bash
# Build + run the tiny-N tensor-pipe microbenchmark on one idle H20 GPU.
# usage: CUDA_VISIBLE_DEVICES=7 bash tests/microbench_tiny_n_wgmma.sh [outdir]
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-/tmp/mb_tiny_n}"
mkdir -p "$OUT"
nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -Xptxas -v -o "$OUT/mb" microbench_tiny_n_wgmma.cu 2>&1 | grep -i "C75\|warn\|error" || true
run() { timeout 300 "$OUT/mb" "$@"; }
echo "# (a) baseline: 2 WG x 2 halves RS m64n8k32 s8, 1 commit/block, wait<1>"
run rs8 2 2
echo "# (a') b304589 form: commit per k step, wait<4>"
run rs8k 2 2
echo "# (b) SS m64n8k32 s8"
run ss8 2 2
echo "# (c) RS/SS m64n16k32 s8 (N padded to 16)"
run rs16 2 2
run ss16 2 2
echo "# (d) warpgroup count sweep (norm = ns per 256x8xK256 equivalent)"
run rs8 1 4
run rs8 1 2
run rs8 1 1
run rs8 2 1
run rs8 3 2
run rs8 3 1
run ss8 1 4
run ss8 3 2
echo "# (a-faithful) kernel 2-buffer loop with per-block RF decode; ss8d = decode->smem + SS; dec = decode only; rs8a = rs8 + independent decode work"
run rs8d 2 2
run dec 2 2
DEC=2 run dec 2 2
DEC=2 run rs8d 2 2
FLAGS=1 run rs8d 2 2
FLAGS=1 run dec 2 2
run rs8a 2 2
FLAGS=1 run rs8a 2 2
echo "# phase offset between the math WGs (FLAGS=8)"
FLAGS=8 run rs8d 2 2
FLAGS=8 run rs8d 3 2
FLAGS=8 run rs8d 3 1
run rs8d 3 1
run rs8d 2 1
FLAGS=8 run rs8d 2 1
echo "# ss8d decomposition: 2 = no fence.proxy, 4 = conflict-free stores"
run ss8d 2 2
FLAGS=2 run ss8d 2 2
FLAGS=4 run ss8d 2 2
FLAGS=6 run ss8d 2 2
FLAGS=12 run ss8d 2 2
echo "# (e) legacy mma.sync m16n8k32 s8, 8 warps"
run imma 2 2
run immal 2 2
echo "# (f) fp8 e4m3 RS m64n8k32 (MXFP4 form)"
run rs8fp8 2 2
run rs8fp8 1 4
