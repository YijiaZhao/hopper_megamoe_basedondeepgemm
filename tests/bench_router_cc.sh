#!/bin/bash
# Build + run the standalone CUDA-core router microkernel (csrc/router_cc_bench.cu) over all
# variants. Usage: bash tests/bench_router_cc.sh [rows] [variants...]   (run on the GPU host)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROWS="${1:-1}"; shift || true
VARS="${*:-a asx b c d1 d2 e f h h2 c78}"
BIN=/tmp/router_cc_bench_$(id -u)
nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -o "$BIN" "$ROOT/csrc/router_cc_bench.cu" 2>&1 | grep -E "registers|spill|error" | sort | uniq -c | head -40
for v in $VARS; do
  timeout 600 "$BIN" --variant "$v" --rows "$ROWS" --iters 100 --stamps 5
done
