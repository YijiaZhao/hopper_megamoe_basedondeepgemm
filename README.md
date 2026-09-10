# H20 MegaMoE Four-API Delivery

This branch delivers four explicit Hopper/H20 MegaMoE APIs and one shared
Fable dynamic-M frontend.  Split and Fused retain independent workspaces,
layouts, schedulers, and ABI contracts.

## Validated APIs

```python
deep_gemm.mxfp4_mega_moe_split
deep_gemm.qoq_mega_moe_split
deep_gemm.mxfp4_mega_moe_fused
deep_gemm.qoq_mega_moe_fused
```

The E2E profiling path for all four APIs uses the same frontend:

```python
deep_gemm.fable_router_quant_topk_frontend(
    hidden_states, router_weight, buffer, quant="mxfp4"  # or "qoq"
)
```

The Fable frontend performs, in one CUDA kernel:

```text
BF16 Router WMMA
+ activation quantization (FP8/K128 or INT8/row)
+ TopK8
+ softmax
```

Legacy GEMV, generic tensor-core, and BF16-GEMM-plus-quant frontends are not
used for the final delivery timelines.

## Validated hardware and software

| Item | Configuration |
|---|---|
| GPU | 8 x NVIDIA H20-3e |
| Architecture | SM90a |
| GPU memory | 143771 MiB per GPU |
| SM count | 78 per GPU |
| Locked SM clock | 1830 MHz |
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 |
| CUDA toolkit | 13.0 |
| Distributed backend | NCCL, 8 ranks |
| Profiler | NVIDIA Nsight Systems, CUDA + NVTX trace, CUDA Graph node tracing |

Validated model shape:

```text
Experts:      384
Local experts: 48 per rank
Hidden:       3072
Intermediate: 1280
TopK:         8
EP:           8
Attention TP: 4 for E2E timelines
Attention DP: 2 for E2E timelines
Global M:     2, 8, 16
```

For a clean-machine Docker setup and end-to-end reproduction procedure, see
[`docs/H20_FOUR_API_DELIVERY.md`](docs/H20_FOUR_API_DELIVERY.md).

Architecture diagram: [`docs/H20_FOUR_API_FUSED_ARCHITECTURE.html`](docs/H20_FOUR_API_FUSED_ARCHITECTURE.html).

### Clean-host Docker quick start

The validated image is pinned by digest:

```text
docker.io/lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7
```

Create the validated container on an 8-GPU H20 host:

```bash
docker pull \
  docker.io/lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7

docker run -d \
  --name four_api_build \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 64g \
  -v /raid:/raid \
  -w /raid/kimi \
  docker.io/lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7 \
  sleep infinity
```

The commands in the next sections run inside this container unless explicitly
marked as host-side. The container is not privileged, so GPU clocks are locked
by `scripts/capture_four_api_h20_timelines_host.sh` on the host.

## Clone and build

```bash
git clone --recursive \
  --branch main \
  https://github.com/YijiaZhao/hopper_megamoe_basedondeepgemm.git
cd hopper_megamoe_basedondeepgemm

export CUDA_HOME=/usr/local/cuda
export DG_CUTLASS_INCLUDE_PATH=$PWD/third-party/cutlass/include
bash develop.sh
```

## Correctness validation

Run on one 8-GPU H20 node:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

# Fable frontend, local M=1/2/4/8/16/32/64, MXFP4 and QoQ
torchrun --standalone --nproc_per_node=8 \
  tests/test_fable_frontend_correctness.py

# Four explicit APIs in one process, without changing DG_W4A8_INT
torchrun --standalone --nproc_per_node=8 \
  tests/test_four_api_smoke.py

# Exact quantized/dequantized references
torchrun --standalone --nproc_per_node=8 \
  tests/test_four_api_correctness.py
```

### Four-API correctness results

| API | Max abs | Mean abs | Min cosine | Mean cosine | Norm ratio |
|---|---:|---:|---:|---:|---:|
| MXFP4 Split | 0.0625 | 0.0047555622 | 0.9999898076 | 0.9999959469 | 0.9999283552 |
| QoQ Split | 0.15625 | 0.033552093 | 0.9999098182 | 0.9999209046 | 0.9970810413 |
| MXFP4 Fused | 0.0625 | 0.0047725799 | 0.9999899864 | 0.9999959469 | 0.9999219179 |
| QoQ Fused | 0.140625 | 0.028386369 | 0.9999374151 | 0.9999402761 | 0.9999618530 |

Acceptance thresholds:

```text
finite output
minimum cosine >= 0.99
0.97 <= norm ratio <= 1.03
```

All four APIs pass.

### Fable frontend correctness and scaling

The frontend test validates TopK selection, selected-logit agreement, softmax
weights, weight sum, and activation quantize/dequantize error.

| Local M | MXFP4 frontend (us) | QoQ frontend (us) |
|---:|---:|---:|
| 1 | 13.775 | 13.737 |
| 2 | 13.738 | 13.751 |
| 4 | 13.777 | 13.822 |
| 8 | 13.963 | 14.066 |
| 16 | 13.498 | 13.588 |
| 32 | 14.459 | 14.848 |
| 64 | 17.854 | 18.365 |

Observed maximum reconstruction error remained below the test thresholds:

```text
MXFP4: 0.033378 < 0.07
QoQ:   0.003937 < 0.005
```

## Nsight Systems timeline matrix

Capture all 24 reports with the recommended host-side wrapper:

```bash
# Run on the host from the repository path. The wrapper locks all eight GPUs
# to 1830 MHz and launches the collector in the validated Docker container.
CONTAINER=four_api_build \
ROOT=$PWD \
OUT=/path/to/empty/output \
LOCK_SM_CLOCK_MHZ=1830 \
bash scripts/capture_four_api_h20_timelines_host.sh
```

To invoke the inner collector directly, first lock clocks on the host, then run
inside the container:

```bash
# Host:
nvidia-smi -lgc 1830,1830

# Container:
OUT=/path/to/empty/output ROOT=$PWD \
LOCK_SM_CLOCK_MHZ=1830 CLOCK_LOCK_MODE=verify \
bash scripts/capture_four_api_h20_timelines.sh
```
Matrix:

```text
2 precisions: MXFP4, QoQ
x 2 scopes: E2E, MegaMoE-only
x 2 backends: Split, Fused
x 3 global token counts: M2, M8, M16
= 24 .nsys-rep files
```

Profiler options:

```text
--trace=cuda,nvtx
--cuda-graph-trace=node
--sample=none
--cpuctxsw=none
```

E2E CUDA Graph contains:

```text
Fable router_quant_topk_kernel
-> Fused MegaMoE kernel
```

or:

```text
Fable router_quant_topk_kernel
-> Split L1 kernel
-> Split L2 kernel
```

The 256 MiB L2 eviction, TP4 ReduceScatter, and TP4 AllGather remain outside
the CUDA Graph.

## Performance aggregation

The table is generated with:

```bash
python3 scripts/verify_four_api_h20_timelines.py /path/to/reports
python3 scripts/summarize_four_api_h20_timelines.py /path/to/reports
python3 scripts/summarize_four_api_h20_last3.py /path/to/reports
```

For every timeline, the delivery number is reported for GPU 0 / rank 0:

1. identify the final three complete MegaMoE executions;
2. measure each complete execution span;
3. report the median of those three spans.

For Split, one complete `Mega span` is measured from its L1 kernel start through
its corresponding L2 kernel end,
including the real inter-kernel gap.  `Target span` in E2E is measured from
Fable frontend start through MegaMoE completion.  It is not a sum of unrelated
kernel statistics and is not aggregated across GPUs.

## Performance results (2026-09-10, locked clock)

Method: `10.6.131.7`, eight H20 GPUs locked at **1830 MHz**,
`scripts/capture_four_api_h20_timelines.sh`, GPU 0 median of the final three
complete spans (see the runnable commands below).  Values are microseconds.  Mega-only Fused is
the customer comparison column (targets M2 < 53, M8 < 61, M16 < 85: all met).

| Precision | M | FE Fused | E2E Fused (FE + Mega) | Mega-only Fused |
|---|---:|---:|---:|---:|
| MXFP4 | 2  | 8.3 (13.9 legacy) | 74.5–96.8   | **42.9** |
| MXFP4 | 4  | 8.3 (13.8 legacy) | 81.5–88.3   | **~49** |
| MXFP4 | 8  | 8.3 (13.9 legacy) | 75.5–89.2   | **56.4–59.1** |
| MXFP4 | 16 | 8.7 (14.0 legacy) | 108.5–117.0 | **74.0–83.4** |
| QOQ   | 2  | 8.7 (14.5 legacy) | 73.5–94.3   | **44.1** |
| QOQ   | 4  | 8.5 (14.6 legacy) | 84.9–87.1   | **48.0** |
| QOQ   | 8  | 8.6 (14.8 legacy) | 80.2–90.3   | **54.9–59.3** |
| QOQ   | 16 | 8.7 (14.9 legacy) | 94.9–109.3  | **74.0** |

FE Fused is the tiny-M Fable frontend (`DG_FE_TINYM=1`, default for m <= 16;
kernel time 6.9 us, nsys span 8.3–8.7 us); the legacy frontend value is in
parentheses.  E2E ranges span captures with different host launch skew.

M2/M4 values are medians over five independent captures (branch tip with
`DG_FP4_STREAMK` default on); M8/M16 are the range over the r4/r5 captures
(`docs/` and `delivery/four_api_fable_timeline_last3_r4_20260909.md`,
`..._r5_20260909.md`).  Ranges reflect host launch skew between the eight
ranks, not kernel variance: the kernel duration on the latest-starting rank
agrees to within 1.5 us across captures.  Adding `dist.barrier()` before each
graph replay in the profiling driver (`DG_PROFILE_HOST_BARRIER=1`) removes
most of that skew (e.g. MXFP4 M8 110.8 -> 57.6 in one A/B).

### How the table is produced (runnable as-is)

```bash
# On 10.6.131.7, inside the four_api_build container, repo at /raid/kimi/dg_dev
cd /raid/kimi/dg_dev && git fetch origin perf/phase-stamps-probe && git reset --hard FETCH_HEAD
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH \
       DG_CUTLASS_INCLUDE_PATH=/raid/kimi/dg_dev/third-party/cutlass/include TORCH_CUDA_ARCH_LIST=9.0a \
       PYTHONPATH=/raid/kimi/dg_dev CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1
bash develop.sh                                   # build the extension (JIT kernels compile on first use)

# correctness gate (exit 0; cos_min MXFP4 >= 0.99998, QoQ >= 0.99992)
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens 2 8 8 16

# one official capture = 8 nsys timelines per M ({e2e,mega} x {split,fused} x {mxfp4,qoq}),
# GPUs must be idle and SM clock locked at 1830 MHz (the script verifies via nvidia-smi dmon)
TOKENS_LIST="2 4 8 16" OUT=/raid/kimi/results/cap_p1 LOCK_SM_CLOCK_MHZ=1830 FORCE=1 \
    bash scripts/capture_four_api_h20_timelines.sh
python3 scripts/summarize_four_api_h20_last3.py /raid/kimi/results/cap_p1     # customer table (GPU0, median of last 3 spans)
python3 scripts/reconcile_nsys_devices.py /raid/kimi/results/cap_p1/mega_fused_mxfp4_M8.nsys-rep   # per-device durations + start skew

# Repeat the capture >= 5 times (cap_p1 .. cap_p5) and take the median across captures per cell:
# a single capture can carry a persistent ~50 us rank-0 launch lead (all 10 rounds), which the
# last-3 median cannot remove.  Optional: DG_PROFILE_HOST_BARRIER=1 adds dist.barrier() before
# each replay in tests/profile_four_api_h20.py and removes most of that skew (not the customer method).

# In-kernel phase breakdown (rank 0 globaltimer stamps; relative comparisons only, not kernel time):
LOG_TAG=_x bash scripts/run_probe.sh mxfp4 2 8 16
```

Knob A/B drivers used for the individual changes live in `scripts/run_*_ab.sh` /
`scripts/run_*_validate.sh` (each waits for idle GPUs and writes under
`/raid/kimi/results/`).

Changes that produced these numbers (all default on unless noted; every knob
has an env override documented in `csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp`):

| Change | Env knob | Effect |
|---|---|---|
| RS register decode for QoQ (int4 -> int8 in the WGMMA A fragments) | – | QoQ inherits the MXFP4 RF pipeline |
| QoQ inline s2 (LiquidGEMM byte-lane dequant, whole-K int32 accumulate, one promote per task) | `DG_FP4_QOQ_INLINE_S2` | QoQ stage -22%, no per-K128 accumulator read-out |
| wgmma pipeline un-serialisation (device CALLs removed, loop-exit `wait<0>`) | – | ptxas no longer wraps every IGMMA in ARRIVE/DEPBAR |
| Fine-grained combine (per-token arrival counters replace NVLink barrier #2) | `DG_FP4_FINE_COMBINE` | -3/-6 us at M8/M16 |
| Push-model dispatch (sender pushes rows + remote ticket; no pull round trip) | `DG_FP4_PUSH_DISPATCH` | first math ~1.5 us after barrier #1 instead of ~7.5 |
| Lean routing (non-zero experts only, no broadcast under push) | `DG_FP4_LEAN_ROUTING` | pre-barrier routing shortened |
| Per-rank DONE flags replace NVLink barrier #1 (push path) | `DG_FP4_PUSH_DONE_FLAGS` | -1 us |
| Deterministic push slots (no remote row ticket; slot = src rank x local token, per-expert slot mask, SM e compacts after DONE) | `DG_FP4_PUSH_DET_SLOTS` (default 0) | push issued ~0.9 us earlier, DONE / first math 0..0.6 us; kernel end within +-1 us (see the host note) |
| QoQ packed-word prefetch inside the RF loop | `DG_FP4_QIS2_PREFETCH_PACKED` | M16 -3..-6 us (probe) |
| stream-K for M <= 4 (units spread over all 78 SMs) | `DG_FP4_STREAMK` (`_MAX_M`) | MXFP4 M2 49.7 -> 42.9 |
| Tiny-M Fable frontend (3 smem stages -> 3 CTAs/SM single wave; 256-thread partial fetch; warp-0 32-bit-key top-8) | `DG_FE_TINYM` | FE 13.8–14.9 -> 8.3–8.7 us (nsys span), bit-identical outputs |

Measured and kept off (documented negative results): 4 K-blocks per stage,
per-M knob sweep, tiny-M CUDA-core GEMV path (`DG_FP4_TINYM`), whole-expert L2
weight prefetch (`DG_FP4_L2_PREFETCH_ALL`), two-layer L1/L2 fusion
(`DG_FP4_FUSE_L1L2`), dynamic combine claim (`DG_FP4_COMBINE_DYNAMIC`), L2
tail split-K, raw-u8 deferred affine dequant.

## Relevant source files

```text
csrc/fable_frontend.cu
csrc/fable_frontend.h
csrc/apis/mega.hpp
csrc/apis/mega_fused.hpp
csrc/jit_kernels/impls/sm90_mxfp4_mega_moe.hpp
csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp
deep_gemm/mega/__init__.py
deep_gemm/mega/fused.py
deep_gemm/include/deep_gemm/impls/sm90_mxfp4_mega_moe.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused_body.inl
tests/test_fable_frontend_correctness.py
tests/test_four_api_smoke.py
tests/test_four_api_correctness.py
tests/profile_four_api_h20.py
scripts/capture_four_api_h20_timelines_host.sh
scripts/capture_four_api_h20_timelines.sh
```

## License

This repository is released under the [MIT License](LICENSE).
