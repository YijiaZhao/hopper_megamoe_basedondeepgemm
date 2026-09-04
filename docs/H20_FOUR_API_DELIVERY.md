# H20 Four-API Customer Reproduction Guide

Updated: 2026-09-04

This is the standalone environment, build, correctness, profiling, and result
reproduction guide for the H20 Four-API delivery.

Interactive architecture diagram:

```text
docs/H20_FOUR_API_FUSED_ARCHITECTURE.html
```

## 1. Validated configuration

| Item | Validated value |
|---|---|
| GPU | 8 x NVIDIA H20-3e, 143771 MiB each |
| Driver | 580.95.05 |
| Architecture | SM90a, 78 SM per GPU |
| Locked SM clock | 1830 MHz |
| OS in container | Ubuntu 24.04 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| CUDA toolkit | 13.0 |
| GCC | 13.3.0 |
| CMake | 3.31.1 |
| Ninja | 1.13.0 |
| Nsight Systems | 2026.4.1.191 |
| Docker image | `docker.io/lmsysorg/sglang:dev` |
| Image digest | `sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7` |

The validated container uses host networking, host IPC, 64 GiB shared memory,
and a same-path `/raid:/raid` bind mount.

## 2. Host prerequisites

The host must have:

- exactly eight visible H20 GPUs;
- NVIDIA driver 580.95.05 or a compatible CUDA 13 driver;
- Docker with NVIDIA Container Toolkit;
- permission to run `nvidia-smi -lgc` on the host;
- an exclusive node with no other GPU processes.

Check the host before starting:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader
```

## 3. Create the validated container

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

The validated container is intentionally not privileged. Clock locking is
therefore performed on the host, not from inside the container.

Verify the software versions:

```bash
docker exec four_api_build bash -lc '
python3 - <<"PY"
import sys, torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CXX11 ABI:", torch.compiled_with_cxx11_abi())
PY
nvcc --version
nsys --version
cmake --version
ninja --version
'
```

## 4. Clone and build

```bash
docker exec -it four_api_build bash

cd /raid/kimi
git clone --recursive \
  --branch main \
  https://github.com/YijiaZhao/hopper_megamoe_basedondeepgemm.git \
  DeepGEMM_four_api_fable_h20_git

cd /raid/kimi/DeepGEMM_four_api_fable_h20_git
git submodule update --init --recursive

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export DG_CUTLASS_INCLUDE_PATH=$PWD/third-party/cutlass/include
export TORCH_CUDA_ARCH_LIST=9.0a
bash develop.sh

python3 - <<'PY'
import deep_gemm
for name in (
    "mxfp4_mega_moe_split",
    "qoq_mega_moe_split",
    "mxfp4_mega_moe_fused",
    "qoq_mega_moe_fused",
    "fable_router_quant_topk_frontend",
):
    assert hasattr(deep_gemm, name), name
print("four APIs and Fable frontend: import OK")
PY
```

## 5. Correctness validation

Inside the container:

```bash
cd /raid/kimi/DeepGEMM_four_api_fable_h20_git
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda
export DG_CUTLASS_INCLUDE_PATH=$PWD/third-party/cutlass/include
unset DG_W4A8_INT DG_W4A8_INT_PRE DG_W4A8_INT_SHADOW

# Fable frontend correctness for MXFP4 and QoQ.
torchrun --standalone --nproc_per_node=8 \
  tests/test_fable_frontend_correctness.py

# Build, four-API smoke test, and exact quantized-reference correctness.
ROOT=$PWD bash scripts/run_four_api_h20_validation.sh all
```

Acceptance gates:

```text
finite output
minimum cosine >= 0.99
0.97 <= norm ratio <= 1.03
```

## 6. Capture the 24 timelines

The recommended entry point is the host-side wrapper. Run it from the repository
path on the host. The repository path must be visible at the same path inside
the container through the `/raid:/raid` mount.

```bash
cd /raid/kimi/DeepGEMM_four_api_fable_h20_git

CONTAINER=four_api_build \
ROOT=$PWD \
OUT=/raid/kimi/results/four_api_h20_locked1830 \
LOCK_SM_CLOCK_MHZ=1830 \
bash scripts/capture_four_api_h20_timelines_host.sh
```

The host wrapper:

1. confirms the container and repository exist;
2. locks all eight GPUs with `nvidia-smi -lgc 1830,1830`;
3. verifies all eight current SM clocks are 1830 MHz;
4. launches the 24-case collector inside the container.

The inner collector:

- refuses a non-empty report directory unless `FORCE=1` is explicitly set;
- checks for competing GPU processes before every case;
- waits for delayed CUDA memory cleanup between cases;
- records every clock sample in `CLOCK_DMON.txt`;
- fails if any recorded SM clock differs from 1830 MHz;
- captures CUDA and NVTX with CUDA Graph node tracing;
- generates SHA256, structural verification, CSV, JSON, and Markdown tables.

The 24-case matrix is:

```text
2 precisions x 2 scopes x 2 backends x 3 M values
= MXFP4/QoQ x E2E/Mega-only x Split/Fused x M2/M8/M16
```

## 7. Verify and regenerate tables

Run these commands inside the container because the report version must not be
newer than the installed Nsight Systems version:

```bash
cd /raid/kimi/DeepGEMM_four_api_fable_h20_git
REPORTS=/raid/kimi/results/four_api_h20_locked1830

python3 scripts/verify_four_api_h20_timelines.py "$REPORTS"
python3 scripts/summarize_four_api_h20_timelines.py "$REPORTS" \
  > "$REPORTS/TIMELINE_TABLE.md"
python3 scripts/summarize_four_api_h20_last3.py "$REPORTS"
sha256sum -c "$REPORTS/SHA256SUMS"
```

Expected structural checks:

```text
24/24 reports valid
E2E: router_quant_topk_kernel present
qoq_quant_topk_kernel absent
Mega-only: no frontend kernel
NCCL outside CUDA Graph
```

## 8. Measurement definition

All published numbers use GPU 0 / rank 0 and the median of the final three
complete executions.

- Fused Mega span: fused MegaMoE kernel start to end.
- Split Mega span: corresponding L1 start through L2 end, including the gap.
- E2E span: Fable frontend start through MegaMoE completion.
- L2 eviction and TP ReduceScatter/AllGather are excluded from E2E target span.

## 9. Output files

A successful output directory contains:

```text
24 x *.nsys-rep
CLOCK_DMON.txt
SHA256SUMS
VERIFY.json
TIMELINE_TABLE.md / .csv / .json
TIMELINE_LAST3.md / .csv / .json
```

Raw `.nsys-rep` files are intentionally excluded from GitHub. The repository
contains the result tables, SHA256 manifest, verification JSON, and scripts.

## 10. Final validated results

The locked-clock result table is shown in the repository root `README.md` and
in:

```text
delivery/four_api_fable_final_summary_20260903.md
delivery/four_api_fable_timeline_last3_20260903.md
```

After profiling, optionally release the host clock lock:

```bash
nvidia-smi -rgc
```
