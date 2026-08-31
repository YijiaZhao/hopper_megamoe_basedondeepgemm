# Customer Quick Start: H20 MegaMoE E384/H3072/I1280/TopK8

This is the self-contained runbook for the validated customer configuration on
8 NVIDIA H20 GPUs. Start from the repository `main` branch. Do not copy files
from older worktrees or experimental Green Context directories.

## 1. Exact source revision

```bash
git clone --recursive https://github.com/YijiaZhao/hopper_megamoe_basedondeepgemm.git
cd hopper_megamoe_basedondeepgemm
git checkout main
git pull --ff-only
```

The clean verification recorded on August 31, 2026 used commits through:

```text
a26b27a docs: record clean-main customer benchmark verification
```

Newer `main` commits are acceptable only after rerunning the tests below.

## 2. Required system and tested container

```text
GPU: 8 x NVIDIA H20-3e (SM90)
Host NVIDIA driver: must support CUDA 13.0 containers
CUDA toolkit: 13.0 (tested: nvcc 13.0.88)
Python: 3.12.3
PyTorch: 2.11.0+cu130
NCCL: 2.28.9
GCC/G++: 13.3
Ninja: 1.13
NCCL-visible topology supporting all 8 GPUs
```

PyTorch symmetric memory is required by the EP8 MegaMoE implementation.

The exact tested image was:

```text
docker.io/lmsysorg/sglang@sha256:08f2f19ffd2522067dadac51716ddce84b13f1e5f3084e1bbd4cb14122998aa8
```

The local container name `ds_new` is not portable. Create a container from the
immutable digest above:

```bash
export DG_REPO=$PWD

docker pull \
  docker.io/lmsysorg/sglang@sha256:08f2f19ffd2522067dadac51716ddce84b13f1e5f3084e1bbd4cb14122998aa8

docker run --name deepgemm-h20-customer --gpus all \
  --network host --shm-size=128g \
  -v "$DG_REPO:/workspace/hopper_megamoe_basedondeepgemm" \
  -w /workspace/hopper_megamoe_basedondeepgemm \
  -d docker.io/lmsysorg/sglang@sha256:08f2f19ffd2522067dadac51716ddce84b13f1e5f3084e1bbd4cb14122998aa8 \
  sleep infinity
```

Verify the container:

```bash
docker exec deepgemm-h20-customer bash -lc '
  python3 --version
  python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version(), torch.cuda.device_count())"
  nvcc --version
  g++ --version | head -1
'
```

Expected key values are Python 3.12, PyTorch 2.11+cu130, CUDA 13.0, and eight
visible H20 GPUs. An equivalent CUDA 13/PyTorch 2.11 image may be used, but the
full validation suite must then be rerun.

## 3. Build inside the container

Ensure the CUTLASS and fmt submodules exist on the host:

```bash
git submodule update --init --recursive
```

Build inside the container:

```bash
docker exec deepgemm-h20-customer bash -lc '
  cd /workspace/hopper_megamoe_basedondeepgemm
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export MAX_JOBS=16
  rm -rf build deep_gemm/_C*.so
  bash develop.sh
'
```

Verify that Python loads this checkout rather than another DeepGEMM install:

```bash
docker exec deepgemm-h20-customer bash -lc '
  cd /workspace/hopper_megamoe_basedondeepgemm
  python3 -c "import deep_gemm; print(deep_gemm.__file__); print(deep_gemm._C.__file__); print(deep_gemm._C.mxfp4_router_quant_topk_split.__doc__); print(deep_gemm._C.qoq_router_quant_topk.__doc__)"
'
```

Both paths must point inside `/workspace/hopper_megamoe_basedondeepgemm`.

## 4. Hardware preparation

Run on the host before benchmarking:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1980
nvidia-smi --query-gpu=index,name,clocks.current.sm,memory.used,utilization.gpu \
  --format=csv,noheader
```

Use all eight idle GPUs. Do not run MXFP4 and QoQ simultaneously because both
benchmarks use the same default distributed rendezvous port.

Enter the container before running the remaining commands:

```bash
docker exec -it deepgemm-h20-customer bash
cd /workspace/hopper_megamoe_basedondeepgemm
```

Common environment inside the container:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DG_SM90_MOE_GREEN_OVERLAP=0
unset DG_SM90_MOE_UNIFIED DG_SM90_MOE_SPLIT_L1_L2
```

## Measurement policy

The two validation modes are intentionally different:

- **All performance measurements use forced balanced routing.** The real
  Router/Quant/TopK frontend is executed and included when it belongs to the
  timed scope, but its TopK IDs and weights are overwritten with the fixed
  balanced pattern before MegaMoE dispatch. This gives exactly eight routed
  assignments per EP rank and removes routing-load variance.
- **All correctness measurements use the normal, unmodified router output.**
  Do not overwrite TopK IDs or weights in correctness runs. Correctness compares
  the TP ReduceScatter path with the AllReduce/reference-token path using the
  same real Router/Quant/TopK results.

Do not add independently measured components to predict E2E latency. In
particular, the historical component table and historical E2E table are not an
additive decomposition because they were collected by separate harnesses/runs,
use max-rank independently, and include different launch/synchronization and
frontend boundaries. `core + communication` also omits Router GEMM,
quantization, TopK, copies, launch gaps, and other timed orchestration overhead.
Only the directly measured full-E2E number is the E2E result.

## 5. Test scaled-MXFP4

```bash
export DG_W4A8_INT=0
export DG_MXFP4_REL_LUT=1
export DG_MXFP4_ABS_SCALE256=1

# Real-router correctness
python3 tests/test_customer_bs1_isl4_tp4_dp2_correctness.py

# MegaMoE core; use rank_mean_us as the reported performance
python3 tests/bench_customer_core_balanced.py \
  --num-processes 8 --attn-tp-size 4 --batches 1 \
  --hidden 3072 --intermediate-hidden 1280 \
  --num-experts 384 --num-topk 8 --block-n 128 \
  --warmup 20 --num-tests 200

# Full E2E; the script reports mean_rank_us across all eight ranks
python3 tests/bench_customer_bs2_tp4_dp2.py \
  --tp-size 4 --batches 4 --balanced-routing \
  --valid-tokens-per-group 4 --warmup 15 --iters 150
```

Expected correctness:

```text
CORRECT mode=scaled-MXFP4 exact=1 max_abs=0 cos_min=0.9999998808
```

## 6. Test QoQ INT4 x INT8

```bash
export DG_W4A8_INT=1
unset DG_MXFP4_REL_LUT DG_MXFP4_ABS_SCALE256

# Real-router correctness
python3 tests/test_customer_bs1_isl4_tp4_dp2_correctness.py

# MegaMoE core; use rank_mean_us as the reported performance
python3 tests/bench_customer_core_balanced.py \
  --num-processes 8 --attn-tp-size 4 --batches 1 \
  --hidden 3072 --intermediate-hidden 1280 \
  --num-experts 384 --num-topk 8 --block-n 128 \
  --warmup 20 --num-tests 200

# Full E2E; the script reports mean_rank_us across all eight ranks
python3 tests/bench_customer_bs2_tp4_dp2.py \
  --tp-size 4 --batches 4 --balanced-routing \
  --valid-tokens-per-group 4 --warmup 15 --iters 150
```

Expected correctness:

```text
CORRECT mode=QoQ exact=1 max_abs=0 cos_min=0.9999998212
```

## 7. What the tests measure

Correctness uses the normal, unmodified real BF16 router output for both paths. It compares:

```text
TP4 ReduceScatter -> Router/Quant/TopK8 -> MegaMoE -> TP4 AllGather
```

against:

```text
TP4 AllReduce -> corresponding token slice
-> same Router/Quant/TopK8 -> same MegaMoE -> TP4 AllGather
```

The deterministic balanced route is used only by performance benchmarks. It is not installed during correctness testing.

Core timing includes only:

```text
EP8 dispatch -> L1 -> SwiGLU/quantized handoff -> L2 -> EP combine
```

E2E timing includes:

```text
TP4 ReduceScatter
-> BF16 Router GEMM
-> activation quantization + TopK8
-> MegaMoE core
-> TP4 AllGather
```

Performance numbers are arithmetic means across all eight ranks. The core test
prints min/mean/max; report `rank_mean_us`. The E2E test directly prints
`mean_rank_us`.

## 8. Clean-main reference results

Fresh build from `main@25681fd` on August 31, 2026:

| precision | correctness | core mean of 3 repeats | E2E mean-rank |
|---|---|---:|---:|
| scaled-MXFP4 | exact=1, max_abs=0 | 115.914 us | 276.089 us |
| QoQ INT4 x INT8 | exact=1, max_abs=0 | 121.352 us | 281.006 us |

Core repeats:

```text
scaled-MXFP4: 108.984, 125.195, 113.564 us
QoQ INT4:    140.911, 107.317, 115.829 us
```

Kernel-only measurements vary between runs, so run at least three repetitions
and retain every result. E2E is the primary production metric.

## 9. Important source locations

```text
csrc/mega_frontend.cu
  mxfp4_quant_topk_from_logits_kernel
  qoq_quant_topk_kernel

csrc/mega_frontend.h
  frontend launcher declarations

csrc/apis/mega.hpp
  mxfp4_router_quant_topk_split
  qoq_router_quant_topk

deep_gemm/include/deep_gemm/impls/sm90_mxfp4_mega_moe.cuh
  SM90 MXFP4/QoQ MegaMoE device implementation

csrc/jit_kernels/impls/sm90_mxfp4_mega_moe.hpp
  SM90 JIT generation and launch path

tests/test_customer_bs1_isl4_tp4_dp2_correctness.py
tests/bench_customer_core_balanced.py
tests/bench_customer_bs2_tp4_dp2.py
```

## 10. Troubleshooting

- `AttributeError: mxfp4_router_quant_topk_split`: Python loaded an old `_C.so`.
  Delete `build/` and `deep_gemm/_C*.so`, rebuild, and verify import paths.
- `misaligned address`: do not use a mixed old worktree or stale JIT cache.
  Re-clone `main`, rebuild, and leave Green/unified experimental switches unset.
- `EADDRINUSE`: another benchmark is using the distributed rendezvous port.
  Wait for it to exit and run the tests sequentially.
- Unexpectedly slow numbers: verify all eight GPUs are idle and clocks are set.
- The test file name `bench_customer_bs2_tp4_dp2.py` is historical; its current
  workload is BS=1 and ISL=4 per DP replica.
