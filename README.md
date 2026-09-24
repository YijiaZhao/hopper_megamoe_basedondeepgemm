# H20 MegaMoE Four-API Delivery

This repository delivers four explicit Hopper/H20 MegaMoE APIs and one shared
Fable dynamic-M frontend.  Split and Fused retain independent workspaces,
layouts, schedulers, and ABI contracts.

## Validated APIs

```python
deep_gemm.mxfp4_mega_moe_split
deep_gemm.qoq_mega_moe_split
deep_gemm.mxfp4_mega_moe_fused
deep_gemm.qoq_mega_moe_fused
```

The E2E path for all four APIs uses the same frontend:

```python
deep_gemm.fable_router_quant_topk_frontend(
    hidden_states, router_weight, buffer, quant="mxfp4"  # or "qoq"
)
```

The Fable frontend performs, in one CUDA kernel:

```text
bf16 router logits (fp32 accumulate, bf16 rounding)
+ activation quantization (FP8 e4m3 per K128 group, or INT8 per row)
+ TopK8 (value desc, expert id asc on ties)
+ softmax over the 8 selected logits
```

The launch shape follows the problem shape (`csrc/fable_frontend.h`, `select_fe_path`):
`m <= 2` rows per rank (H 3072, E 384, top-8; the M = 2 .. 16 pipeline on 8 ranks) run the
CUDA-core K-split router `router_cc_lean_kernel` (77 CTAs x 5 experts x 4 K-part warps, weights
streamed straight into registers, one 32-bit key per (token, expert)); with
`DG_FE_SELECT_IN_MEGA=1` the frontend stops after the keys and the fused MegaMoE prologue selects
the top-8 + softmax. `m <= 16` rows run the mma.sync m16n8k16 router (experts on the MMA M
dimension, fragment-permuted weights) on the 96-CTA grid; every other shape runs the WMMA router.

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
Global M:     2, 4, 8, 16
```

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
on the host (`nvidia-smi -lgc 1830,1830`; `scripts/capture_four_api_h20_timelines_host.sh`
does it before launching the collector in the container).

## Clone and build

```bash
git clone --recursive \
  --branch main \
  https://github.com/YijiaZhao/hopper_megamoe_basedondeepgemm.git
cd hopper_megamoe_basedondeepgemm

export CUDA_HOME=/usr/local/cuda
export DG_CUTLASS_INCLUDE_PATH=$PWD/third-party/cutlass/include
bash develop.sh          # builds the extension; the fused MegaMoE JIT-compiles on first use
```

## Correctness verification

All gates run on one 8-GPU H20 node with the library defaults (`DG_FE_SELECT_IN_MEGA=1` where
the pipeline uses it) on the same kernels the performance table was captured with.
Details and per-cell tables: [`docs/fe5_correctness.md`](docs/fe5_correctness.md).

* **End-to-end vs a pure-torch real-MoE reference** (`tests/test_four_api_correctness.py
  --frontend fe --reference torch-moe`, 8 ranks): the reference computes the bf16 router GEMM
  in fp32 -> bf16-rounded logits -> top-8 (value desc, index asc) -> fp32 softmax over the 8
  logits -> torch per-token quantisation of x -> dequantised expert GEMMs (both layers,
  exact-quantised weights, SwiGLU, clamp) -> weighted combine, with the frontend's REAL routing
  of random hidden rows; none of the frontend / MegaMoE kernels is in the reference.
  Pass criteria: finite output, `cos_min >= 0.99`, `0.97 <= norm ratio <= 1.03`, per-(token,
  slot) `--slot-check` clean on 8/8 ranks. Result on this tree (T = 1 / 2 / 8 / 16 / 32 rows per
  rank, both fused APIs): frontend top-8 index sets equal the torch router's on every token,
  softmax weights within 1 fp32 ulp, MXFP4 x byte-identical to the torch cast (QoQ: <= 1 int8
  step on ~0.5 elements per 1000, an exact-.5 rounding-tie difference, x_sf identical);
  `cos_min` MXFP4 >= 0.99992, QoQ >= 0.99992, norm ratio 0.9998 .. 1.00004.
* **Forced-balanced routing** (`DG_FE_FORCE_BALANCED=1`): the frontend runs, then its routing
  is replaced by the balanced assignment the profiler's `DG_PROFILE_FORCE_BALANCED=1` memcpy
  writes (`expert = s * 48 + (g + 7 s) % 48`, weight 1/8, unrouted inactive rows) and the torch
  reference routes with it: all cells PASS (`cos_min` MXFP4 >= 0.99996, QoQ >= 0.99992).
* **50-seed sweep** (14 shapes x {SELECT_IN_MEGA 0, 1} x {normal, balanced}, 292 800 evaluated
  tokens): 0 failures, 0 top-8 disagreements, 0 Mega-prologue vs python-decode differences;
  worst `cos_min` MXFP4 0.99975 (M = 128 / 256), QoQ 0.99990.
* **Zero-row gate** (`tests/test_four_api_correctness.py --zero-rows K [--zero-ranks ...]`, 8 ranks, both
  fused APIs, `DG_FE_SELECT_IN_MEGA` 0 and 1): all-zero hidden rows (the E2E padding rows: T = 1 with
  zero rows on the ranks that own no token at M2 / M4, T = 2 with one zero and one real row per rank,
  T = 8 swapab path) -- the frontend leaves every zero row unrouted on the cc path, the kernel output of
  every zero row is exactly 0, and x / x_sf / the routing / y of the other rows are bit-identical between
  the pipeline with `DG_FE_ZERO_ROW_UNROUTED` 0 and 1 (5 seeds per cell; T = 2 with the split-K tail
  `DG_FP4_SPLITK_L1=1`: <= 2 bf16 ulps on 2 of 24 576 elements in one QoQ seed, bit-identical with the
  tail off). `tests/fe_dump_compare.py --zero-rows 1` knob 0 vs 1, 64 seeds: differences only in the
  routing outputs (`topk_idx`, `topk_weights`, keys) of the zero rows, `x` / `x_sf` identical.
* **Select-in-Mega gate** (`tests/test_select_in_mega.py`, 8 ranks, 50 seeds, M 2 / 16, both
  quants): 0 topk mismatches, y bit-identical between select-in-frontend and select-in-Mega.
* **Frontend output layout gate** (`tests/fe_dump_compare.py`): the frontend runs alone for
  64 seeds x rows {1, 2} x {mxfp4, qoq} and every buffer the fused Mega consumes (`x`, `x_sf`,
  `topk_idx`, `topk_weights`, the ticket area, the select-in-Mega `x` / `x_sf`, the compact
  key array) is byte-compared between two builds or env configurations, padding rows included;
  `--ref-torch` compares `x` / `x_sf` with the torch per-token casts.

Commands (inside the container, repo root; 8 GPUs unless noted):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1
# frontend standalone, local M = 1/2/4/8/16/32/64, both quants
torchrun --standalone --nproc_per_node=8 tests/test_fable_frontend_correctness.py
# four explicit APIs in one process; exact quantised references (synthetic balanced routing)
torchrun --standalone --nproc_per_node=8 tests/test_four_api_smoke.py
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py
# frontend + fused Mega vs the pure-torch MoE reference, T rows per rank (T = 1 2 8 16 32)
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --reference torch-moe --tokens 1
DG_FE_FORCE_BALANCED=1 torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --reference torch-moe --tokens 1 --seeds 50
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --tokens 32 --hot-rows 12 --slot-check
# zero-row gate: M2-like owner layout (zero rows on ranks 1,2,3,5,6,7), mixed rows, swapab path
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused \
    --frontend fe --reference torch-moe --tokens 1 --zero-rows 1 --zero-ranks 1,2,3,5,6,7 --seeds 5
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused \
    --frontend fe --reference torch-moe --tokens 2 --zero-rows 1 --seeds 5 --zero-rows-y-ulp 2
python3 tests/fe_dump_compare.py --env-a DG_FE_ZERO_ROW_UNROUTED=0 --env-b DG_FE_ZERO_ROW_UNROUTED=1 --zero-rows 1 --seeds 64  # 1 GPU
bash tests/fe5_gates.sh                                     # the gate set above, T = 1 2 8 16 32 + slot checks
OUT=/raid/kimi/results/fe5c bash tests/fe5c_sweep.sh ; python3 scripts/summarize_fe5c_sweep.py /raid/kimi/results/fe5c/*.log
python3 tests/fe_dump_compare.py --root-a <reference checkout> --root-b . --seeds 64 --ref-torch   # 1 GPU
```

## Performance results

Per-platform result documents (one table + measurement method + reproduce each):

| Platform | Kernels | Mega-only, forced-balanced, GPU 0, µs at global M = 2 / 4 / 8 / 16 | Document |
|---|---|---|---|
| 8x H20-3e (SM90, 1830 MHz) | MXFP4 / QoQ fused (this repo) | 38.8 / 47.4 / 56.7 / 76.2 (MXFP4), 37.1 / 45.2 / 54.0 / 72.6 (QoQ) | [docs/H20_MEGAMOE_RESULTS.md](docs/H20_MEGAMOE_RESULTS.md) |
| 8x B200 (SM100, 1965 MHz) | W-MXFP4xA-FP8 / W4A4 MXFP4 / W4A4 NVFP4 (`sm100_b200/`, on AichenF DeepGEMM `megamoe_nvfp4_dev`) | 39.4 / 43.0 / 49.2 / 55.8 (FP8 act), 35.4 / 39.9 / 45.3 / 53.9 (MXFP4), 36.1 / 40.2 / 45.6 / 51.6 (NVFP4) | [docs/B200_MEGAMOE_RESULTS.md](docs/B200_MEGAMOE_RESULTS.md) |

Same model scale for both (E384 / 48 local, H3072, I1280, top-8, EP8) and the same forced-balanced routing; both columns are the
kernel span on GPU 0, median of the last 3 of 30 streamed replays (nsys on H20, torch-profiler kernel durations on B200).

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
deep_gemm/include/deep_gemm/impls/fable_frontend_device.cuh
deep_gemm/include/deep_gemm/impls/fable_cc_select.cuh
deep_gemm/include/deep_gemm/impls/sm90_mxfp4_mega_moe.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused_body.inl
tests/test_fable_frontend_correctness.py
tests/test_four_api_smoke.py
tests/test_four_api_correctness.py
tests/test_select_in_mega.py
tests/fe_dump_compare.py
tests/profile_four_api_h20.py
tests/fe5_campaign_streamed.sh
tests/fe5_summarize_campaign.py
scripts/capture_four_api_h20_timelines_host.sh
scripts/capture_four_api_h20_timelines.sh
scripts/summarize_four_api_h20_last3.py
```

## License

This repository is released under the [MIT License](LICENSE).
