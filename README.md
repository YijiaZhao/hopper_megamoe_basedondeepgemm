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
| Locked SM clock | 1980 MHz |
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

Capture all 24 reports:

```bash
# Run inside an exclusive 8-GPU environment after building the extension.
# The script refuses a non-empty output directory and checks GPU memory before
# and after every case.
OUT=/path/to/output ROOT=$PWD \
  bash scripts/capture_four_api_h20_timelines.sh

# Replace an existing output set intentionally:
FORCE=1 OUT=/path/to/output ROOT=$PWD \
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
```

For every target:

1. take the final three executions on each GPU;
2. take their median per GPU;
3. take the arithmetic mean across all eight GPUs.

For Split, `Mega span` is measured from L1 kernel start through L2 kernel end,
including the real inter-kernel gap.  `Target span` in E2E is measured from
Fable frontend start through MegaMoE completion.  It is not a sum of unrelated
kernel averages.

The final table, raw values, report verification, and SHA256 manifest are kept
under `delivery/`:

```text
delivery/four_api_fable_final_summary_20260903.md
delivery/four_api_fable_timeline_table_20260903.md
delivery/four_api_fable_timeline_table_20260903.csv
delivery/four_api_fable_timeline_table_20260903.json
delivery/four_api_fable_timeline_verify_20260903.json
delivery/four_api_fable_timeline_sha256_20260903.txt
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
```

## License

This repository is released under the [MIT License](LICENSE).
