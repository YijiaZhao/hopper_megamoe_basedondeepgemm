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
  --branch delivery/four-api-fable-h20-20260903 \
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

## Final locked-clock performance results

The following values were captured on `10.6.131.7` with all eight H20 GPUs
explicitly locked at **1830 MHz**. All values are microseconds and are the
GPU 0 / rank 0 median of the final three complete spans.

| Precision | M | FE Fused | FE Split | E2E Mega Fused | E2E Mega Split | E2E Fused | E2E Split | Mega-only Fused | Mega-only Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 13.760 | 13.856 | 86.624 | 107.520 | **100.640** | **121.440** | 107.808 | 57.152 |
| MXFP4 | 8 | 13.632 | 13.792 | 91.520 | 125.568 | **105.440** | **139.424** | 87.584 | 105.088 |
| MXFP4 | 16 | 13.696 | 13.856 | 133.024 | 147.424 | **147.008** | **161.088** | 115.232 | 128.416 |
| QOQ | 2 | 14.272 | 14.400 | 92.896 | 87.040 | **107.584** | **101.632** | 64.192 | 50.400 |
| QOQ | 8 | 14.432 | 14.432 | 95.456 | 84.352 | **110.144** | **98.880** | 92.608 | 81.760 |
| QOQ | 16 | 14.720 | 14.400 | 138.784 | 125.120 | **153.504** | **139.968** | 123.712 | 114.112 |

Validation summary:

```text
24/24 NSYS reports structurally valid
4968/4968 clock samples at 1830 MHz
250 ms GPU-process audit violations: 0
qoq_quant_topk_kernel calls: 0
E2E reports using router_quant_topk_kernel: 12/12
```

The final table, raw values, report verification, and SHA256 manifest are kept
under `delivery/`:

```text
delivery/four_api_fable_final_summary_20260903.md
delivery/four_api_fable_timeline_table_20260903.md
delivery/four_api_fable_timeline_table_20260903.csv
delivery/four_api_fable_timeline_table_20260903.json
delivery/four_api_fable_timeline_verify_20260903.json
delivery/four_api_fable_timeline_sha256_20260903.txt
delivery/four_api_fable_timeline_last3_20260903.md
delivery/four_api_fable_timeline_last3_20260903.csv
delivery/four_api_fable_timeline_last3_20260903.json
```

Raw `.nsys-rep` files are not committed to Git because of their size.  They are
packaged separately in the customer delivery directory.

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
