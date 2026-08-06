# MegaMoE BF16 Router Frontend + TP Results (2026-08-06)

## Latest results (read this first)

### Kernel Factory fused frontend

| Frontend | Geomean latency | PyTorch baseline | Speedup | Correctness |
|---|---:|---:|---:|---|
| scaled-MXFP4: BF16 Router + FP8 K128 Quant + TopK6 | **32.32 us** | 98.11 us | **3.04x** | 7/7 PASS |
| QoQ: BF16 Router + whole-row INT8 Quant + TopK6 | **31.14 us** | 128.60 us | **4.13x** | 7/7 bitwise PASS |

### Natural router distribution: final BF16 output including TP AllGather

| M/rank | scaled-MXFP4 TP2/DP4 | scaled-MXFP4 TP4/DP2 | QoQ TP2/DP4 | QoQ TP4/DP2 |
|---:|---:|---:|---:|---:|
| 4  | **194.5** | 216.7 | 223.6 | **202.7** |
| 8  | **219.5** | **220.6** | 224.3 | 227.1 |
| 16 | 242.8 | 262.9 | **233.4** | **236.5** |
| 32 | **264.1** | **270.5** | 276.3 | 281.4 |
| 44 | **278.4** | **286.8** | 294.1 | 300.5 |
| 48 | **280.5** | **288.1** | 297.0 | 302.6 |
| 64 | **304.9** | **314.5** | 338.8 | 346.8 |

### Forced balanced routing: final BF16 output including TP AllGather

| M/rank | scaled-MXFP4 TP2/DP4 | scaled-MXFP4 TP4/DP2 | QoQ TP2/DP4 | QoQ TP4/DP2 |
|---:|---:|---:|---:|---:|
| 4  | 221.5 | **182.1** | 233.3 | 215.8 |
| 8  | 221.9 | **205.7** | 227.3 | 210.4 |
| 16 | **228.5** | 232.6 | 229.4 | **232.4** |
| 32 | **254.3** | **259.6** | 264.6 | 269.6 |
| 44 | **269.1** | **275.4** | 282.2 | 285.5 |
| 48 | **290.8** | **287.5** | 307.0 | 302.0 |
| 64 | 314.5 | **305.9** | **312.6** | 321.8 |

All times are microseconds and include fused Frontend, MegaMoE L1/L2, EP
dispatch/combine, and final TP token AllGather. The balanced benchmark includes
two conservative GPU copies used to override routing metadata.

This document records the current SM90/H20 implementation and the measurements
used to validate the fused BF16-input MegaMoE paths on branch
`w_int4_a_int8_qserve_and_w_mxfp4_a_fp8`.

## Scope and semantics

End-to-end API:

```text
rank-local BF16 hidden
  -> fused BF16 Router + Quant + TopK6 frontend
  -> MegaMoE L1 grouped GEMM
  -> MegaMoE L2 + EP combine
  -> optional TP token AllGather
  -> global BF16 output
```

The attention-TP router weight is replicated, but tokens are DP-sharded: every
physical rank is an independent EP dispatch source. Broadcast is not valid for
distinct token shards. The recommended final collective is token-axis
AllGather. A reduce-to-root + broadcast implementation is retained only as a
performance comparison.

Precision modes:

- **scaled MXFP4**: W=OCP E2M1, weight scale=E8M0 per output-channel/K32,
  A=FP8 E4M3 with per-token/K128 FP32 scale. The experimental fast path uses
  `DG_MXFP4_REL_LUT=1 DG_MXFP4_ABS_SCALE256=1`.
- **QoQ W4A8**: W=canonical QoQ uint4 + asymmetric zero point, integer `s2`
  per K128, BF16 `s1` per output channel; A=INT8 with one whole-row per-token
  scale repeated into the 32 K128 scale slots. Decode is SHIFTXOR/shift-mask +
  zero-point subtraction and INT8 WGMMA.

## Fused frontend kernels

Both frontends use Kernel Factory generated SM90 CUDA kernels with an 8-CTA
cluster, K512 TMA/WGMMA router pipeline, distributed TopK6, and shape-specific
quantization ownership.

| Frontend | Geomean latency | Baseline | Speedup | Correctness |
|---|---:|---:|---:|---|
| MXFP4 BF16 Router + FP8 K128 Quant + TopK6 | 32.32 us | 98.11 us | 3.04x | 7/7 PASS |
| QoQ BF16 Router + whole-row INT8 Quant + TopK6 | 31.14 us | 128.60 us | 4.13x | 7/7 bitwise PASS |

Kernel Factory campaigns:

- MXFP4: `m6zr8v0bp11she60kg5qferxjg`, winner
  `6e18baec1bfd33a9decbfa1d42e41b5cafd3c6cb9eabf0b5042c8836422fc600`.
- QoQ: `k5299y7hs13c1fys3e6y0rfdd0`, current winner
  `228b72df9a77e2216f9cd21d81af278f695c66874859afd37fa1761725e9db03`.

## Final TP AllGather performance

H20-3e, 1830 MHz, 8 ranks / EP8, H=4096, I=2048, E=128, TopK=6,
BN128. Times are CUDA-event wall time from rank-local BF16 input through final
global BF16 output and include Router, Quant, TopK, L1/L2, EP combine, and TP
AllGather. PDL is disabled.

### Natural router distribution

| M/rank | scaled-MXFP4 TP2/DP4 | scaled-MXFP4 TP4/DP2 | QoQ TP2/DP4 | QoQ TP4/DP2 |
|---:|---:|---:|---:|---:|
| 4  | 194.5 | 216.7 | 223.6 | 202.7 |
| 8  | 219.5 | 220.6 | 224.3 | 227.1 |
| 16 | 242.8 | 262.9 | 233.4 | 236.5 |
| 32 | 264.1 | 270.5 | 276.3 | 281.4 |
| 44 | 278.4 | 286.8 | 294.1 | 300.5 |
| 48 | 280.5 | 288.1 | 297.0 | 302.6 |
| 64 | 304.9 | 314.5 | 338.8 | 346.8 |

### Forced balanced routing

This benchmark still executes the real fused frontend, then overwrites TopK
IDs/weights with a deterministic balanced pattern before dispatch. The two GPU
copies are included, so the values are conservative and intended for
compute-path comparison rather than deployment prediction.

| M/rank | scaled-MXFP4 TP2/DP4 | scaled-MXFP4 TP4/DP2 | QoQ TP2/DP4 | QoQ TP4/DP2 |
|---:|---:|---:|---:|---:|
| 4  | 221.5 | 182.1 | 233.3 | 215.8 |
| 8  | 221.9 | 205.7 | 227.3 | 210.4 |
| 16 | 228.5 | 232.6 | 229.4 | 232.4 |
| 32 | 254.3 | 259.6 | 264.6 | 269.6 |
| 44 | 269.1 | 275.4 | 282.2 | 285.5 |
| 48 | 290.8 | 287.5 | 307.0 | 302.0 |
| 64 | 314.5 | 305.9 | 312.6 | 321.8 |

## TP collective comparison

AllGather is the production recommendation. Reduce-to-root + broadcast produces
the same global token layout by placing each local shard in a disjoint global
slot, reducing to TP rank 0, then broadcasting the global tensor. It is slower
at every measured QoQ point and almost every MXFP4 point.

Example M64:

| Mode | MXFP4 TP2 | MXFP4 TP4 | QoQ TP2 | QoQ TP4 |
|---|---:|---:|---:|---:|
| AllGather | 416.3 us (standard MXFP4) | 423.7 us | 368.1 us (pre-fused QoQ frontend) | 377.9 us |
| Reduce + Broadcast | 468.9 us | 511.2 us | 416.5 us | 450.6 us |

The current optimized QoQ AllGather result is 338.8/346.8 us for TP2/TP4.

## Correctness

- QoQ frontend: exact INT8 bytes, repeated whole-row scale, TopK IDs, and TopK
  weights for M in `{4,8,16,32,44,48,64}`.
- MXFP4 frontend: exact FP8 bytes and scales; TopK IDs exact; weights pass the
  BF16 GEMM tolerance gate.
- scaled-MXFP4 8-rank exact-dequant reference gate: 7/7 PASS,
  `cosine_min=0.944639`, `cosine_mean=0.999489`, norm ratio
  `[0.999353, 1.000512]`.
- TP AllGather transports BF16 output without arithmetic modification.

## Core implementation notes

- BN128 swap-AB is active for all measured M values. Runtime selects
  `N_SWAP` from `{8,16,32,64}` using the actual expert token count.
- Small M uses a 4-stage RF pipeline; larger M uses 6 stages to trade SMEM
  footprint against TMA latency.
- QoQ uses register SHIFTXOR decode and one integer coefficient group per K128.
- Standard MXFP4 converts E8M0 with exponent-bit construction during
  accumulator promotion. The scaled path folds the E8M0 exponent into decoded
  FP8 before WGMMA and removes a shared x256 factor at promotion.
- Whole-kernel PDL is disabled: early launch of the persistent L1 grid creates
  starvation/slow modes, while a tail trigger provides no useful overlap.

## Commands

```bash
# QoQ TP2 + AllGather
DG_W4A8_INT=1 DG_WALL_BENCH=1 \
python tests/bench_mxfp4_mega_moe_sm90.py \
  --num-processes 8 --num-experts 128 --attn-tp-size 2 \
  --tp-combine-allgather --batches 4 8 16 32 44 48 64 \
  --block-n 128 --bf16-e2e --warmup 10 --num-tests 100

# scaled-MXFP4 TP4 + AllGather
DG_MXFP4_REL_LUT=1 DG_MXFP4_ABS_SCALE256=1 DG_WALL_BENCH=1 \
python tests/bench_mxfp4_mega_moe_sm90.py \
  --num-processes 8 --num-experts 128 --attn-tp-size 4 \
  --tp-combine-allgather --batches 4 8 16 32 44 48 64 \
  --block-n 128 --bf16-e2e --warmup 10 --num-tests 100

# Force balanced routing for comparative profiling
DG_BALANCED_ROUTING=1 <precision env> DG_WALL_BENCH=1 python \
  tests/bench_mxfp4_mega_moe_sm90.py <same args>
```

## Remaining production caveat

`DG_MXFP4_ABS_SCALE256=1` currently assumes the supported E8M0 code range used
by the measured checkpoint-like weights. Before enabling it unconditionally,
add a prepack-time range check and fall back to relative-LUT/standard promotion
for out-of-range E8M0 codes.
