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

## Performance results (2026-09-12, round 5, locked clock) -- single source of truth

Machine and method: `10.6.131.8`, eight H20-3e locked at **1830 MHz** (`nvidia-smi -lgc 1830,1830`, the capture
script verifies), container `fe5c_build` (same image as `four_api_build`), branch `perf/phase-stamps-probe` kernels
`eba23b1` (build `fee8931`, 06:54 UTC), one session 06:54-08:39 UTC, `scripts/capture_four_api_h20_timelines.sh`
(nsys, `--cuda-graph-trace=node`), **GPU 0, median of the last 3 graph replays**, then the median over passes.
Values are microseconds; M = global tokens over the 8 ranks (M2/M4/M8 = 1 row per rank, M16 = 2 rows), both quants,
fused backend (`mxfp4|qoq_mega_moe_fused`), pipeline knobs `DG_FE_SELECT_IN_MEGA=1 DG_FE_CC_LEAN=1` (library defaults
except SELECT_IN_MEGA, which the profiling drivers default to 1). Older tables (2026-09-04 / 09-10 captures on
10.6.131.7) are superseded by this section; the per-change history stays in the knob table below and in `docs/`.

### a. Correctness verification

Everything below is on the same kernels the tables were captured with (`DG_FE_CC_LEAN=1` and `=0`). Details, logs and
per-cell tables: `docs/fe5_correctness.md`, `docs/fe_cc_round5.md` section 4.

* **FE output layout, byte gate** (`tests/fe_dump_compare.py`): the FE runs alone under two env configurations (one
  child process each, the knobs are process-static) for 64 seeds x rows {1, 2} x {mxfp4, qoq} = 256 cells and every
  buffer the fused Mega consumes is byte-compared after a sentinel pre-fill: `x` [rows 0..3 incl. padding, 3072] bytes
  (mxfp4 e4m3 per K128 group; qoq int8 whole row in the same e4m3 tensor), `x_sf` [4, 24] fp32 (mxfp4 amax/448 per
  K128; qoq amax/127 replicated), `topk_idx` [4, 8] int64, `topk_weights` [4, 8] fp32, the 256 B ticket area, the
  select-in-Mega launch's `x` / `x_sf`, the 4 KB compact key array at workspace byte 65792 (token t at + t * 1536) and
  the ticket area after it. Result `DG_FE_CC_LEAN=0` vs `=1`: `FE_DUMP_COMPARE cells=256 failing_cells=0`, all 9 buffers
  identical in all 256 cells, padding rows included; `--ref-torch`: mxfp4 x / x_sf byte-identical to the torch
  per-token cast in every cell. (`tests/fe5_ident.py`, the same comparison over topk / keys / ticket / x / x_sf, also
  PASS 256/256 for LEAN, pruned select and both together.)
* **End-to-end vs a pure-torch real-MoE reference** (`tests/test_four_api_correctness.py --frontend fe --reference
  torch-moe`, 8 ranks): the reference computes the bf16 router GEMM in fp32 -> bf16-rounded logits -> top-8 (value
  desc, index asc) -> fp32 softmax over the 8 logits -> torch per-token quantisation of x -> dequantised expert GEMMs
  (both layers, exact-quantised weights, SwiGLU, clamp) -> weighted combine, with the FE's REAL routing of random
  hidden rows; none of our FE / Mega kernels is in the reference. FE vs torch router: top-8 index sets equal on every
  token (8/8 .. 256/256 per launch; 292 800 / 292 800 tokens in the sweep), softmax weights within 1 fp32 ulp
  (6e-8 .. 2.4e-7), mxfp4 x identical, qoq x differs by <= 1 quantisation step on ~0.5 elements per 1000 (see the
  note below). y vs reference (single seed, .8, T = 1 / 2 / 8 / 16 / 32 rows per rank, both LEAN settings identical to
  the last printed digit): mxfp4 cos_min 0.99999 / 0.99999 / 0.99992 / 0.99992 / 0.99992, qoq 0.99993 x3 / 0.99992 x2,
  norm ratio 0.9998-1.00004; per-(token, slot) `--slot-check` clean on 8/8 ranks (T=32, and the balanced
  `--tokens 32 --hot-rows 12 --slot-check`).
* **Forced-balanced routing verified the same way**: `DG_FE_FORCE_BALANCED=1` in the correctness test applies the same
  override the profiler uses (FE runs, then the routing is replaced by the Mega-only assignment `expert = s * 48 +
  (g + 7 s) % 48`, weight 1/8 each, unrouted inactive rows) and the torch reference routes with it: all cells PASS
  (sweep rows "balanced" in `docs/fe5_correctness.md` 4.2, cos_min mxfp4 >= 0.99996, qoq >= 0.99992).
* **ComputeLab 50-seed sweep** (job 4252043, 8 x H20-3e): 14 shapes x 3 configurations (LEAN 0/1 x SELECT_IN_MEGA
  0/1) = 42 launches, 50 seeds each, 292 800 evaluated tokens, 0 failures, 0 top-8 disagreements, 0 SELECT_IN_MEGA
  prologue/python-decode differences; worst cos_min mxfp4 0.99993, qoq 0.99992 (M = 2 .. 256).
* **Known benign difference**: the FE's QoQ int8 quantisation rounds `v * (1/scale)` where the torch cast rounds
  `x / scale`; on exact .5 ties this is 1 int8 step on a handful of elements per row (223 bytes of 8 x 3072 per launch
  at M8), x_sf identical, same for LEAN 0 and 1, within the reference tolerance. mxfp4 has no such difference.
* Gates that must stay green (all PASS on this tip): `tests/test_select_in_mega.py` (8 ranks, 50 seeds, M 2 / 16,
  both quants: 0 topk mismatches, y bit-identical between select-in-FE and select-in-Mega), `tests/test_frontend_fe78.py
  --mma cc --seeds 40 --rows 1 2` (cc / lean vs the legacy WMMA router: 0 top-8 set mismatches, 0 x mismatches).

Commands (8 GPUs unless noted; inside the container, repo root):
```bash
python3 tests/fe_dump_compare.py --env-a DG_FE_CC_LEAN=0 --env-b DG_FE_CC_LEAN=1 --seeds 64 --ref-torch      # 1 GPU
python3 tests/fe5_ident.py --save /tmp/ref.pt ; DG_FE_CC_LEAN=1 python3 tests/fe5_ident.py --ref /tmp/ref.pt  # 1 GPU
torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --reference torch-moe --tokens 1   # T = 1 2 8 16 32
DG_FE_FORCE_BALANCED=1 torchrun --standalone --nproc_per_node=8 tests/test_four_api_correctness.py \
    --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --reference torch-moe --tokens 1 --seeds 50
LEANS="0 1" OUT=/raid/kimi/results/fe5c bash tests/fe5c_sweep.sh ; python3 scripts/summarize_fe5c_sweep.py /raid/kimi/results/fe5c
bash tests/fe5_gates.sh                                                                                    # the gate set above
```

### b. E2E (FE + MegaMoE in one CUDA graph), normal routing (the FE's real top-8)

**Streamed method** (`DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30`, `tests/fe5_campaign_streamed.sh`): the 30 replays of
a case are enqueued back-to-back on the stream (no per-iteration `torch.cuda.synchronize()` / `dist.barrier()`), so
the ranks lock to each other through the on-stream reduce-scatter / all-gather instead of re-skewing at every host
round trip; GPU-0 median of the last 3 replays, median over 3 passes. `skew` = inter-rank spread of the Mega kernel
start (max - min over the 8 devices, median over passes of the per-pass max of the last 3 replays).

| Precision | M | FE (lean) | Mega (in graph) | E2E (FE start -> Mega end) | Mega-start skew |
|---|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 2.6 | 60.9 | **63.8** | 2.1 |
| MXFP4 | 4 | 2.8 | 60.1 | **63.2** | 1.7 |
| MXFP4 | 8 | 2.6 | 65.2 | **68.1** | 2.4 |
| MXFP4 | 16 | 2.8 | 78.2 | **81.3** | 1.4 |
| QOQ | 2 | 2.6 | 60.4 | **63.5** | 50.0 |
| QOQ | 4 | 2.7 | 60.2 | **63.2** | 6.7 |
| QOQ | 8 | 2.7 | 63.4 | **66.3** | 1.5 |
| QOQ | 16 | 3.0 | 76.6 | **80.0** | 1.6 |

**Plain customer method** (per-iteration host sync + barrier, `tests/fe5_campaign.sh`, 8 replays per case, GPU-0
median of the last 3, median over 5 passes; base = `DG_FE_CC_LEAN=0`, lean = `1`):

| Precision | M | FE base | FE lean | Mega base | Mega lean | E2E base | E2E lean | skew base / lean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 3.8 | 2.5 | 69.2 | 66.2 | 73.7 | 69.0 | 12.9 / 19.3 |
| MXFP4 | 4 | 4.1 | 2.5 | 69.5 | 90.2 | 73.4 | 93.0 | 13.8 / 50.8 |
| MXFP4 | 8 | 4.0 | 2.5 | 72.2 | 76.4 | 76.3 | 79.3 | 14.9 / 20.7 |
| MXFP4 | 16 | 4.0 | 2.8 | 89.4 | 92.2 | 93.5 | 95.4 | 20.9 / 13.7 |
| QOQ | 2 | 4.2 | 2.7 | 66.0 | 66.3 | 70.5 | 69.3 | 14.7 / 11.9 |
| QOQ | 4 | 4.3 | 2.7 | 72.0 | 69.2 | 76.6 | 72.2 | 23.3 / 17.6 |
| QOQ | 8 | 4.1 | 2.7 | 68.6 | 71.9 | 73.1 | 74.8 | 17.0 / 15.3 |
| QOQ | 16 | 4.4 | 2.8 | 91.4 | 95.3 | 95.7 | 98.5 | 14.3 / 17.5 |

Why the two methods differ: the fused Mega kernels of the 8 ranks END within ~1 us of each other (the first in-kernel
NVLink barrier aligns them), so a rank's Mega span is `common end - its own start` = kernel work + that rank's head
start over the slowest rank. With a host sync + barrier between replays the ranks re-skew by 10-60 us at every launch
(the `skew` column), and GPU 0 -- usually among the first to launch -- absorbs it inside its Mega span; the E2E /
Mega columns of the plain method therefore measure launch skew as much as kernel work, and their base-vs-lean
differences (+-5..20 us, in both directions) are that skew's noise. Streamed, the skew collapses to 1-2.5 us in 12
of 16 cells (M2 and qoq M4 stay looser: with one active token per TP half the collectives carry little data) and the
Mega column converges to the kernel work. Per-rank decomposition of one plain replay (`scripts/decompose_e2e_skew.py`,
balanced MXFP4 M2, base, pass 1, replay -3; times relative to the earliest FE start):

| device | FE start | FE end | FE span | gap -> Mega | memcpy node | Mega start | Mega end | Mega span |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| GPU0 | 0.0 | 4.0 | 4.0 | 1.3 | 4.1-5.2 | 5.3 | 60.8 | 55.5 |
| GPU1 | 5.4 | 9.4 | 4.1 | 1.4 | 9.5-10.7 | 10.8 | 60.7 | 50.0 |
| GPU2 | 4.1 | 8.7 | 4.6 | 1.2 | 8.7-9.8 | 9.8 | 60.6 | 50.7 |
| GPU3 | 11.7 | 16.1 | 4.4 | 1.2 | 16.2-17.2 | 17.3 | 60.5 | 43.2 |
| GPU4 | 14.7 | 19.0 | 4.3 | 1.3 | 19.0-20.3 | 20.3 | 61.7 | 41.4 |
| GPU5 | 11.9 | 16.3 | 4.5 | 1.4 | 16.4-17.6 | 17.7 | 60.7 | 43.0 |
| GPU6 | 5.0 | 9.1 | 4.2 | 1.2 | 9.2-10.2 | 10.3 | 60.7 | 50.4 |
| GPU7 | 6.5 | 10.8 | 4.3 | 1.3 | 10.9-12.1 | 12.2 | 60.4 | 48.3 |
| skew (max - min) | 14.7 | 15.0 | | | | 15.0 | 1.3 | |

GPU 0's Mega span 55.5 = the latest rank's 41.4 (the kernel work, = the Mega-only column) + 14.1 head start; the FE
is 4.0-4.6 us on every rank; the FE end -> Mega start gap is 0.3 us with normal routing and 1.2-1.5 us in balanced
mode (the 1.1-1.2 us memcpy override node); there are no other nodes in the graph. The same replay streamed:
FE 2.7-2.8, gap 1.2-1.4, Mega-start skew 0.9 / 2.8 / 6.5 us, GPU 0 Mega span 40.2 / 41.9 / 40.6 vs streamed
Mega-only 38.2-38.5. Full per-rank tables for M2 (both quants, normal / balanced, plain / streamed):
`~/Downloads/h20_fused_official/fe5/decomp/`.

### c. E2E with forced-balanced routing (`DG_PROFILE_FORCE_BALANCED=1`)

The FE runs exactly as in (b); one graph memcpy node (`cudaMemcpyAsync`, no kernel, 1.1-1.2 us between the FE and
Mega kernels, inside the E2E span, outside both kernel spans) then overwrites its routing with the Mega-only scope's
balanced assignment: active row t of rank r (global token g = r * rows + t) -> expert `s * 48 + (g + 7 s) % 48` for
slot s = 0..7, one route to every EP rank, **uniform weights 1/8**; inactive owner-layout rows (M < 8) unrouted. With
`DG_FE_SELECT_IN_MEGA=1` the compact key array is overwritten (chosen experts logit 1.0, others 0.0 -> the Mega prologue
selects them in slot order, softmax of eight equal logits = exactly 0.125; all-zero keys = unrouted row), otherwise
`topk_idx` / `topk_weights` directly. Same layout as (b):

Streamed (3 passes):

| Precision | M | FE (lean) | Mega (in graph) | E2E | Mega-start skew |
|---|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 2.7 | 40.7 | **44.7** | 15.8 |
| MXFP4 | 4 | 2.7 | 50.1 | **54.0** | 2.1 |
| MXFP4 | 8 | 2.7 | 58.3 | **62.3** | 1.8 |
| MXFP4 | 16 | 2.8 | 76.8 | **80.8** | 1.1 |
| QOQ | 2 | 2.8 | 42.0 | **46.1** | 15.4 |
| QOQ | 4 | 2.8 | 50.2 | **54.4** | 2.0 |
| QOQ | 8 | 2.8 | 56.5 | **60.6** | 1.4 |
| QOQ | 16 | 3.0 | 76.0 | **80.5** | 1.5 |

Plain customer method (5 passes):

| Precision | M | FE base | FE lean | Mega base | Mega lean | E2E base | E2E lean | skew base / lean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 4.1 | 2.6 | 49.4 | 60.0 | 54.8 | 64.2 | 43.9 / 62.1 |
| MXFP4 | 4 | 4.1 | 2.5 | 50.8 | 63.5 | 56.0 | 67.4 | 13.4 / 20.1 |
| MXFP4 | 8 | 4.1 | 2.5 | 76.8 | 64.1 | 82.2 | 68.0 | 24.2 / 16.8 |
| MXFP4 | 16 | 4.3 | 2.9 | 84.7 | 82.6 | 90.3 | 86.9 | 27.2 / 24.8 |
| QOQ | 2 | 4.3 | 2.8 | 47.3 | 43.9 | 52.7 | 48.2 | 14.5 / 14.7 |
| QOQ | 4 | 4.3 | 2.8 | 58.2 | 53.1 | 64.1 | 57.2 | 12.8 / 18.6 |
| QOQ | 8 | 4.3 | 2.8 | 66.6 | 56.0 | 72.5 | 60.1 | 20.2 / 23.0 |
| QOQ | 16 | 4.3 | 3.0 | 91.2 | 81.4 | 96.6 | 85.8 | 21.1 / 12.6 |

Streamed and balanced, E2E = FE + 1.3 us (memcpy node + launch gap) + Mega, and Mega = Mega-only + 1-3 us of
residual skew (e.g. MXFP4 M2: 44.7 = 2.7 + 1.3 + 40.7 with Mega-only 38.3). The difference between (b) and (c) in the
Mega column (60-65 vs 41-58 us at M2-M8) is the real cost of the FE's actual, unbalanced top-8 routing against the
uniform assignment -- not skew.

### d. Mega-only, balanced routing (the customer comparison column)

Mega-only scope (no FE; balanced assignment, uniform 1/8 weights, as in the 2026-09-04 delivery), fused backend.
Targets: M2 < 53, M8 < 61, M16 < 85 us.

| Precision | M | plain customer method (5 passes) | streamed (1 pass) | target | status |
|---|---:|---:|---:|---:|---|
| MXFP4 | 2 | 45.0 | 38.3 | < 53 | met |
| MXFP4 | 4 | 54.3 | 46.9 | – | |
| MXFP4 | 8 | 60.5 | 56.8 | < 61 | met (plain 60.5) |
| MXFP4 | 16 | 78.7 | 74.4 | < 85 | met |
| QOQ | 2 | 48.7 | 37.4 | < 53 | met |
| QOQ | 4 | 50.8 | 46.4 | – | |
| QOQ | 8 | 56.6 | 53.3 | < 61 | met |
| QOQ | 16 | 78.5 | 73.0 | < 85 | met |

The Mega-only scope has the same launch-skew exposure as (b): its plain-method Mega-start skews were 7-36 us per
replay (one 1530 us outlier in qoq M2 pass 1), which is why its plain numbers sit 4-11 us above the streamed ones.

### e. FE kernel (`router_quant_topk_kernel` -> `router_cc_lean_kernel`)

`DG_FE_CC_LEAN=1` (default since round 5; `0` = the previous generic kernel) compiles the CUDA-core cc router
(77 CTAs x 5 experts x 4 K-part warps, weights straight into registers, 384 compact keys per token, spare CTA
quantises) as its own entry point: 2.7 K SASS instructions instead of the 11.9 K-instruction generic instantiation
whose router role sat 73 KB into the kernel behind a far branch and was fetched cold from DRAM after every L2 flush /
Mega weight stream (NCU top stall `no_instruction` 6.9 -> 1.7 warps per issue cycle, warp instructions -44 %, SM
elapsed cycles -26 %, registers 96 -> 64; `~/Downloads/h20_fused_official/ncu/ncu_fe5/KEY_TABLE.md`). The hot path is
the first code of the kernel, the quant CTA and the knob-0 last-arriver select are `__noinline__` cold paths, weights
are issued before activations, activations are converted to fp32 once in the weight-latency shadow (1-instruction
bf16 -> fp32), row 1 runs only when m == 2, and the spare CTA quantises both rows concurrently (320 threads and a named
barrier per row; the sequential QoQ row quant had been the kernel's critical path at 2 rows). Numerics unchanged
(same fma chain / butterfly / K-part order, one bf16 rounding) -> bit-identical outputs (section a).

FE kernel span, plain customer method, 5-pass medians (every cell 5/5 passes; base and lean pass distributions never
overlap; standalone single-GPU nsys spans 2.94/2.91 -> 2.50/2.50 rows 1, 3.17/3.46 -> 2.88/2.91 rows 2 in
`docs/fe_cc_round5.md`):

| Precision | M | normal base -> lean | balanced base -> lean | streamed lean (normal / balanced) |
|---|---:|---:|---:|---:|
| MXFP4 | 2 | 3.8 -> 2.5 | 4.1 -> 2.6 | 2.6 / 2.7 |
| MXFP4 | 4 | 4.1 -> 2.5 | 4.1 -> 2.5 | 2.8 / 2.7 |
| MXFP4 | 8 | 4.0 -> 2.5 | 4.1 -> 2.5 | 2.6 / 2.7 |
| MXFP4 | 16 | 4.0 -> 2.8 | 4.3 -> 2.9 | 2.8 / 2.8 |
| QOQ | 2 | 4.2 -> 2.7 | 4.3 -> 2.8 | 2.6 / 2.8 |
| QOQ | 4 | 4.3 -> 2.7 | 4.3 -> 2.8 | 2.7 / 2.8 |
| QOQ | 8 | 4.1 -> 2.7 | 4.3 -> 2.8 | 2.7 / 2.8 |
| QOQ | 16 | 4.4 -> 2.8 | 4.3 -> 3.0 | 3.0 / 3.0 |

Negative (kept as a knob, default off): `DG_FE_CC_SELECT=pruned`, a two-level threshold top-8 (lane maxes ranked by
an all-gather, bound = 8th lane max, fast path when no lane holds two keys >= bound, else <= 3 candidates per lane +
8 redux rounds; bit-identical): knob-0 FE span 4.48 -> 4.51 us (rows 1), 4.64 -> 4.99 (rows 2); the single-warp
select is issue-bound and, in the pipeline (SELECT_IN_MEGA=1), off the critical path anyway. Earlier FE negatives
(78 full-K wmma / FMA, 78 x 2 K-parts, tc16, ccfp8, PDL, L2 persist alone, radix select, FE-into-Mega fusion) are in
the negatives list below and in `docs/fe_cc_round4.md`.

### f. Reproduce

Requirements: 8 idle H20 (the capture script waits for <= 64 MiB used per GPU and no compute process), SM clock
locked at 1830 MHz (`nvidia-smi -lgc 1830,1830`; `LOCK_SM_CLOCK_MHZ=1830 CLOCK_LOCK_MODE=verify` in the script),
the extension built from this branch (`bash tests/build_ccrouter.sh` or `develop.sh`; the fused Mega JIT-compiles on
first use), `DG_FE_SELECT_IN_MEGA=1` in the pipeline (the profiling driver's default). Inside the container, repo root:
```bash
# (b)+(c)+(d) plain customer method: 5 passes x {base, lean} x {normal, balanced} E2E + Mega-only, interleaved
bash tests/fe5_campaign.sh 5 /raid/kimi/results/fe5/cap            # ~65 min
# (b)+(c)+(d) streamed: 3 passes, lean, normal + balanced E2E, then one Mega-only pass
bash tests/fe5_campaign_streamed.sh 3 /raid/kimi/results/fe5/cap_streamed   # ~16 min
# tables (median over passes of each capture's GPU-0 median-of-last-3; skew columns; skew-filtered medians)
python3 tests/fe5_summarize_campaign.py /raid/kimi/results/fe5/cap 20
python3 tests/fe5_summarize_campaign.py /raid/kimi/results/fe5/cap_streamed 20
# one capture set by hand (what the campaign scripts call; env selects build / routing / method)
DG_FE_CC_LEAN=1 DG_PROFILE_FORCE_BALANCED=0 DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30 DG_FE_SELECT_IN_MEGA=1 \
  OUT=/raid/kimi/results/x SCOPES=e2e BACKENDS=fused QUANTS="mxfp4 qoq" TOKENS_LIST="2 4 8 16" FORCE=1 \
  bash scripts/capture_four_api_h20_timelines.sh
SCOPES=e2e BACKENDS=fused python3 scripts/summarize_four_api_h20_last3.py /raid/kimi/results/x   # GPU-0 last-3 medians + skew
python3 scripts/decompose_e2e_skew.py /raid/kimi/results/x/e2e_fused_mxfp4_M2.nsys-rep \
  --mega-only /raid/kimi/results/x_mega/mega_fused_mxfp4_M2.nsys-rep     # per-rank FE / gap / memcpy / Mega + skew
# FE standalone A/B + identity + NCU (single GPU)
FE5_GPU=7 bash tests/fe5_chain.sh /raid/kimi/results/fe5 ; NCU_FE_GPU=7 bash tests/ncu_fe5.sh   # ncu needs docker exec --privileged
```
Knobs used by these tables: `DG_FE_CC_LEAN` (default 1; 0 = generic cc44 kernel), `DG_FE_SELECT_IN_MEGA` (library
default 0, pipeline drivers 1), `DG_PROFILE_STREAMED` (0 = customer method; 1 = back-to-back replays) with
`DG_PROFILE_ITERS` (8; 30 in the streamed tables), `DG_PROFILE_FORCE_BALANCED` (profiler: FE + memcpy override of the
routing), `DG_FE_FORCE_BALANCED` (correctness test: the same override, reference follows it), `DG_PROFILE_HOST_BARRIER`
(1 = `dist.barrier()` before each replay in the plain method; a 3-pass lean-only run with it is in
`docs/fe_cc_round5.md` 3d for reference, not part of the tables above), `DG_FE_CC_SELECT` (`insert` | `pruned` | `redux`).
Raw artefacts of this section: `~/Downloads/h20_fused_official/fe5/` (per-capture `TIMELINE_LAST3.*`, campaign logs,
gate log, decompositions, NCU reports).

### Legacy full-matrix capture (split + fused, 2026-09-10 flow; commands still valid)

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
| QoQ packed-word prefetch inside the RF loop | `DG_FP4_QIS2_PREFETCH_PACKED` | M16 -3..-6 us (probe) |
| stream-K for M <= 4 (units spread over all 78 SMs) | `DG_FP4_STREAMK` (`_MAX_M`) | MXFP4 M2 49.7 -> 42.9 |
| Wide L1 tasks (BN=512, 1 K-block/stage, 3-way tail split) for M >= 16 | `DG_FP4_L1_BN` (`DG_FP4_BN512_MIN_M`) | M16 Mega-only 5-capture medians MXFP4 84.1 -> 82.6, QoQ 80.4 -> 77.4 |
| Tiny-M Fable frontend (3 smem stages -> 3 CTAs/SM single wave; 256-thread partial fetch; warp-0 32-bit-key top-8) | `DG_FE_TINYM` | FE 13.8–14.9 -> 8.3–8.7 us (nsys span), bit-identical outputs |
| FE swapped-operand router MMA: experts on the MMA M dimension (mma.sync m16n8k16 bf16, tokens pad to 8), weight A fragments ld.global.nc straight into registers (whole K-part in flight, no smem ring / TMA), activations staged once in smem; router weights permuted ONCE into A-fragment order so one warp load = one contiguous 512 B = 4 full lines (`deep_gemm.fable_router_weight_fragment_layout`, cached per weight tensor or pass `wlayout='pre'`) | `DG_FE_TINYM_MMA=auto` (default) = `swapab` on the 96 grid for rows > 2 (rows <= 2: cc row below; `wmma` = legacy), `DG_FE_ROUTER_WLAYOUT=fragment` (default) `|row|pre` (the permuted layout is applied only where the legacy-96 tiny-M swapab kernel reads it: m <= 16, top-8, h 3072; tinym=0 / m > 16 / full-K launches get row-major weights) | ON by default: customer method (nsys `router_quant_topk_kernel` span, 1830 MHz, M=8 = 1 row/rank, mxfp4/qoq) swapab+fragment 7.68/7.52 us vs wmma 8.00/8.74 (same session A/B); H20-3e kernel end (stamps, rows 1/2 x mxfp4/qoq) 96x4 wmma 6.91 -> swapab row 6.66 -> swapab+fragment 6.14 us (-0.77); 78 full-K swapab 6.40-6.91 (vs 7.94-8.45 full-K wmma); NCU 96/rows 1: L2 read requests 49.1k -> 37.7k (-23%) -> 19.0k (-61%), read sectors 113k -> 77k (-32%), LSU inst 89k -> 29k; top-8 sets identical on 54k rows for both swapab row and fragment (0 mismatches, 0 weight flips), 8-rank cos_min unchanged; standalone CUDA-event FE (~19.4-20.9 us) is CPU-launch-bound and shows no signal |
| FE SM-count grid: full-K router CTAs (H20 77 x 5 experts, TMA row pieces + WMMA or CUDA-core FMA), 32-bit-key slots as flags, one streaming-merge CTA, quant on router CTAs, K-parts 1/2/4 | `DG_FE_TINYM_GRID=96|auto|N`, `DG_FE_TINYM_MMA=wmma|fma`, `DG_FE_TINYM_KPARTS`, `DG_FE_FULLK_1PERSM` | kept OFF (default 96): kernel end 96x4 6.9 us vs 78 full-K wmma 7.4–7.9 / fma 7.4–7.7, 78x2 wmma 8.2 / fma 9.7, 97x4 wmma 9.2; nsys FE span 8.66 (96) vs 8.77–9.15 (auto) us; top-8 sets identical on 27k rows |
| FE CUDA-core K-split router (full-K grid, m <= 2): CTA = 5 experts x 4 warps (640 threads), one warp = one expert x one quarter of K, all weight chunks `ld.global.nc` straight into registers as the first instructions; last-arriving router CTA (`atom.acq_rel.gpu` ticket) reads the 616 key slots and does top-8 + softmax (no polling merger CTA); the spare CTA quantises the rows | `DG_FE_TINYM_GRID=auto DG_FE_TINYM_MMA=cc` (`cc6` = 6 warps/expert; `DG_FE_CC_MERGE=ticket|poll`, `DG_FE_CC_SELECT=insert|redux`) | kept OFF pending the customer bench: H20-3e kernel end (stamps, rows 1/2 x mxfp4/qoq) 96x4 wmma 6.91/7.17/6.66/6.91 -> cc 3.84 in all four cells (-2.8..-3.3 us; defaults: relaxed atomic ticket, compact 384-key array, router weights in the persisting L2 set-aside; steps: slots cold 4.61-4.86 -> persist 4.35-4.61 -> compact keys 3.84); chain rows 1: logits 1.79, ticket 2.82, keys read 3.07, select 3.58, written 3.84; `ccfp8` (e4m3 weights) 4.35 us and 18% top-8 set flips = rejected; microkernel (`csrc/router_cc_bench.cu`) all 384 logits at 2.05 us, the SM's outstanding-request capacity bounds the 2.36 MB stream (TMA bulk variants 2.75-3.4 us, slower); top-8 sets identical on 4200 row-evaluations (2 bf16 flips), x/x_sf bit-identical; 8-rank fused correctness cos_min 0.99999 (mxfp4) / 0.99993 (qoq); ROUND 4 (docs/fe_cc_round4.md): the 3.84 holds only for back-to-back launches (10 fresh processes x 100 launches: 3.58-3.84 median, all 4 cells); any ms-scale host gap between launches -> 5.4-5.9 (straggling weight loads + relaxed-ticket wait), inside the FE+Mega graph 6.4; NCU key table in h20_fused_official/ncu/ncu_fe_cc (issue 23 %, occupancy 31 %, top stall no_instruction, L2 persist not keeping the matrix resident); `DG_FE_SELECT_IN_MEGA=1` (FE ends after the keys, Mega prologue selects, topk bit-identical): FE kernel 2.05-2.30 standalone / 3.3 in-graph, FE+Mega graph -1..-4 us in all 6 cells (knob, default 0) |
| Split-K tail tasks of a wave-scheduled launch use the split-K (not the stream-K) reduction slots | – (fix) | real, unbalanced routing at M <= 8 with >= 8 active local experts: cos_min 0.67 -> 0.99999 (`tests/test_four_api_correctness.py --frontend fe`) |
| QoQ inline s2 gated to launches with <= 8 possible rows per local expert (`num_tokens x num_ranks <= 8`) | `DG_FP4_QOQ_INLINE_S2` (default 1, now host-gated) | fix: with > 8 rows on one expert (two BM8 pool blocks) the inline-s2 L1 path returns wrong partials for the second block's rows; found by `--frontend fe` T=32 (qoq cos_min 0.0007), reproduced with default routing T=64/128 (cos_min < 0), `DG_FP4_QOQ_INLINE_S2=0` passes; no perf change at M <= 8 (1 row/rank), M16 loses the inline promote |
| FE cc lean entry point (round 5, docs/fe_cc_round5.md): `router_cc_lean_kernel`, the cc44 router compiled as its own kernel (2.7 K SASS instructions vs 11.9 K for the generic instantiation whose router role sat 73 KB into the kernel), weights issued before activations, activations converted once in the weight-latency shadow, 1-instruction bf16 -> fp32, row 1 only when m == 2, spare CTA quantises the two rows concurrently; bit-identical outputs | `DG_FE_CC_LEAN` (default 1; 0 = generic kernel) | ON: standalone nsys span (pipeline config) mxfp4/qoq rows 1 2.94/2.91 -> 2.50/2.50 us, rows 2 3.17/3.46 -> 2.88/2.91; NCU no_instruction stall 6.9 -> 1.7 warps/issue-cycle, instructions -44 %, elapsed cycles -26 %; customer method: see the perf section |
| FE two-level threshold top-8 select (round 5): lane maxes ranked by an all-gather, bound = 8th lane max, fast path when no lane holds two keys >= bound, else <= 3 candidates per lane + 8 redux rounds | `DG_FE_CC_SELECT=pruned` (default `insert`) | kept OFF: knob-0 FE span 4.48 -> 4.51 us rows 1, 4.64 -> 4.99 rows 2 (the single-warp select is issue-bound; the fast path covers ~45 % of random tokens); in the pipeline the select is off the critical path anyway; bit-identical |
| Profiling: streamed replays (no per-iteration host sync / barrier inside the MEASURE loop), replay count | `DG_PROFILE_STREAMED` (default 0 = customer method), `DG_PROFILE_ITERS` (default 8) | measurement only: inter-rank Mega-start skew 12-62 us -> 1-2.5 us (section b) |
| Profiling: forced-balanced routing in the E2E scope (FE runs, one graph memcpy overrides its routing with the Mega-only assignment, uniform 1/8 weights) | `DG_PROFILE_FORCE_BALANCED` (default 0) | measurement only (section c); `DG_FE_FORCE_BALANCED` = the same override in `tests/test_four_api_correctness.py` |
| Profiling: host barrier before each replay (plain method) | `DG_PROFILE_HOST_BARRIER` (default 0) | measurement only; docs/fe_cc_round5.md 3d |
| FE select-in-Mega: the cc router stops after its 384 keys per token, the fused Mega prologue selects top-8 + softmax (`deep_gemm/impls/fable_cc_select.cuh`) | `DG_FE_SELECT_IN_MEGA` (library default 0; pipeline drivers default 1 for the fused backend) | ON in the pipeline: FE ~3.1-3.3 us in-graph (6.4 without), FE+Mega -1.1..-4.2 us in all 6 cells; 8-rank 50 seeds 0 mismatches, y bit-identical; round-4 fixes: key-array view offset, `ld.global.cg` for prologue-written topk |

Measured and kept off (documented negative results): 4 K-blocks per stage,
per-M knob sweep, tiny-M CUDA-core GEMV path (`DG_FP4_TINYM`), whole-expert L2
weight prefetch (`DG_FP4_L2_PREFETCH_ALL`), two-layer L1/L2 fusion
(`DG_FP4_FUSE_L1L2`), dynamic combine claim (`DG_FP4_COMBINE_DYNAMIC`), L2
tail split-K, all-task L1/L2 split-K at M16, wide L2 tasks (BN=512), third math warpgroup (`DG_FP4_MATH_WGS=3`), deterministic push slots (`DG_FP4_PUSH_DET_SLOTS`), FE router-weight L2 persistence and FE->Mega PDL, raw-u8 deferred affine dequant, FE SM-count router grid (`DG_FE_TINYM_GRID=auto`: 78 full-K CTAs, also 78x2 / 97x4 K-parts, WMMA and CUDA-core FMA, 2 CTAs/SM grids 117/156 -- every variant is bounded by the ~2.5 us it takes one SM to get ~36 KB of router weights in flight after the L2 flush, so the balanced 31 KB/SM grid loses to the legacy 26 KB/SM + 19 doubled SMs; stamp tables in `scripts/run_fe78*.sh` outputs),
Fable frontend fused into the MegaMoE kernel (`DG_FP4_FUSE_FE`, docs/fe_into_mega_design.md: bit-identical outputs, E2E +4.8..+7.6 us at M=2/8/16),
round-2 CUDA-core FE router (slot keys) as the default (kernel-end stamps 6.91 -> 4.35 us, but nsys kernel span 8.9-9.7 vs 7.5-7.7 us swapab under the customer method; superseded by round 3 = compact keys, which is the default for rows <= 2),
FE two-level threshold top-8 select (`DG_FE_CC_SELECT=pruned`, round 5: +0.03..+0.35 us on the knob-0 FE span, off the critical path in the pipeline; docs/fe_cc_round5.md).

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
