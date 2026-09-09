#!/usr/bin/env bash
# Build + run the tiny-N tensor-pipe microbenchmark on one idle H20 GPU.
# usage: CUDA_VISIBLE_DEVICES=7 bash tests/microbench_tiny_n_wgmma.sh [outdir] [section...]
# sections: pipe decode knobs offload sass (default: all)
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-/tmp/mb_tiny_n}"; shift || true
SECTIONS="${*:-pipe decode knobs offload sass}"
mkdir -p "$OUT"
nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -Xptxas -v -o "$OUT/mb" microbench_tiny_n_wgmma.cu 2> "$OUT/build.log" || { cat "$OUT/build.log"; exit 1; }
grep -i "C75\|warn\|error" "$OUT/build.log" || true
run() { timeout 300 "$OUT/mb" "$@"; }
for sec in $SECTIONS; do case "$sec" in
pipe)
echo "# (a) baseline pipe: 2 WG x 2 halves RS m64n8k32 s8, constant A, 1 commit/block, wait<1>"
run rs8 2 2
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
echo "# (e) legacy mma.sync m16n8k32 s8, 8 warps"
run imma 2 2
run immal 2 2
echo "# (f) fp8 e4m3 RS m64n8k32 (MXFP4 form)"
run rs8fp8 2 2
run rs8fp8 1 4
;;
decode)
echo "# (a-faithful) kernel 2-buffer loop with per-block RF decode (rs8d); dec = decode only; rs8a = constant-A wgmma + independent decode work"
run rs8d 2 2
run dec 2 2
run rs8a 2 2
DEC=2 run rs8d 2 2
DEC=2 run dec 2 2
DEC=2 run rs8a 2 2
FLAGS=1 run rs8d 2 2
FLAGS=1 run dec 2 2
FLAGS=1 run rs8a 2 2
echo "# 3-buffer software pipeline (rs8t): two groups in flight while decoding"
run rs8t 2 2
DEC=2 run rs8t 2 2
FLAGS=1 run rs8t 2 2
run rs8t 3 2
run rs8t 1 4
echo "# WG count / phase offset (FLAGS=8) for the decode loop"
run rs8d 3 2
run rs8d 3 1
run rs8d 2 1
run rs8d 1 4
FLAGS=8 run rs8d 2 2
FLAGS=8 run rs8t 2 2
echo "# ss8d decomposition: 2 = no fence.proxy, 4 = conflict-free stores"
run ss8d 2 2
FLAGS=2 run ss8d 2 2
FLAGS=4 run ss8d 2 2
FLAGS=6 run ss8d 2 2
;;
knobs)
echo "# kernel knobs on the rs8d loop: 32 = prefetch packed LDS before wait<1>, 64 = raw-u8 RS wgmma + deferred affine, 96 = both"
run rs8d 2 2
FLAGS=32 run rs8d 2 2
FLAGS=64 run rs8d 2 2
FLAGS=96 run rs8d 2 2
FLAGS=1 run rs8d 2 2
FLAGS=65 run rs8d 2 2
FLAGS=96 run rs8d 3 2
FLAGS=96 run rs8d 3 1
;;
offload)
echo "# reference: kernel form rs8d with phase stamps"
run rs8d 2 2
echo "# offloaded decode + SS wgmma (ss8p): writers = WG0 (4 warps) | FLAGS=16 all 12 warps; FLAGS+2 = no proxy fence"
run ss8p 2 2
FLAGS=16 run ss8p 2 2
FLAGS=2 run ss8p 2 2
FLAGS=18 run ss8p 2 2
DEC=2 run ss8p 2 2
DEC=2 FLAGS=16 run ss8p 2 2
echo "# 3-tile offload, raw u8 codes + deferred per-row affine (ss8u); FLAGS=2 no proxy fence"
run ss8u 2 2
FLAGS=2 run ss8u 2 2
DEC=2 run ss8u 2 2
;;
sass)
echo "# SASS excerpt: rs8d<2,2,0> and rs8t<2,2,0> main loop (tensor / warpgroup / LDS / branch instructions only)"
for fn in _Z12bench_kernelILi8ELi2ELi2ELi0EEvPyS_Piii _Z12bench_kernelILi8ELi2ELi2ELi96EEvPyS_Piii; do
  cuobjdump -sass -fun "$fn" "$OUT/mb" > "$OUT/$fn.sass" 2>/dev/null || true
  echo "## $fn: $(grep -c IGMMA "$OUT/$fn.sass") IGMMA, $(grep -c 'WARPGROUP.ARRIVE' "$OUT/$fn.sass") ARRIVE, $(grep -c 'WARPGROUP.DEPBAR' "$OUT/$fn.sass") DEPBAR, $(grep -c 'LDS' "$OUT/$fn.sass") LDS, $(wc -l < "$OUT/$fn.sass") lines"
  echo "   per-block instruction mix in the loop (opcode histogram):"; awk '/WARPGROUP.DEPBAR/{c++} c==2' "$OUT/$fn.sass" | grep -o '^ */\*[0-9a-f]*\*/ *[@!P0-9 ]*[A-Z][A-Z0-9_.]*' | sed 's/.*\*\/ *//; s/^@!*P[0-9] *//' | sort | uniq -c | sort -rn | head -14 | awk '{printf "      %5d %s\n", $1, $2}'
  grep -o 'IGMMA[^;]*;\|WARPGROUP[^;]*;\|LDS[^;]*;\|STS[^;]*;\|BRA [^;]*;\|DEPBAR[^;]*;\|BAR[^;]*;\|FENCE[^;]*;' "$OUT/$fn.sass" | sed 's/  */ /g' | awk '{print "   " $0}' | head -60
done
;;
esac; done
