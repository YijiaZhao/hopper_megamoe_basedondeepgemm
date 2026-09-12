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
| MXFP4 | 2 | 4.1 (7.8 swapab, 8.3 wmma, 13.9 legacy) | 68.4–119.4 | **42.9** |
| MXFP4 | 4  | 8.3 (13.8 legacy) | 81.5–88.3   | **~49** |
| MXFP4 | 8 | 4.2 (7.6 swapab, 8.3 wmma, 13.9 legacy) | 83.6–95.9 | **56.4–59.1** |
| MXFP4 | 16 | 4.1 (8.0 swapab, 8.7 wmma, 14.0 legacy) | 95.8–113.5 | **74.0–82.6** |
| QOQ | 2 | 4.0 (7.6 swapab, 8.7 wmma, 14.5 legacy) | 66.9–116.2 | **44.1** |
| QOQ   | 4  | 8.5 (14.6 legacy) | 84.9–87.1   | **48.0** |
| QOQ | 8 | 4.1 (7.7 swapab, 8.6 wmma, 14.8 legacy) | 77.5–169.3 | **54.9–59.3** |
| QOQ | 16 | 3.9 (7.9 swapab, 8.7 wmma, 14.9 legacy) | 90.6–97.2 | **74.0–77.4** |

Round 5 (below): with the lean cc entry point (`DG_FE_CC_LEAN=1`, default) the FE Fused column is 2.5-3.0 us at
M2/4/8/16 in both quants (5-pass medians, 10.6.131.8), from 3.8-4.4 for the same build with the knob off.
FE Fused is the tiny-M Fable frontend (`DG_FE_TINYM=1`, default for m <= 16). Library default
`DG_FE_TINYM_MMA=auto`: rows <= 2 per rank (every customer point: M2/M8 = 1 row, M16 = 2 rows) run
the round-3 CUDA-core K-split router (`cc`, SM-count grid, compact 384-key array, kernel-end
stamps 3.84 us); rows > 2 run the swapped-operand MMA router + fragment weight layout (`swapab`,
96 grid). In the PIPELINE (FE immediately followed by the fused Mega: `tests/profile_four_api_h20.py`
E2E scope, `tests/bench_frontend_tinym.py` FE+Mega) the FE additionally runs with
`DG_FE_SELECT_IN_MEGA=1`: the router CTAs end after storing their 384 keys per token and the fused
Mega prologue does the top-8 + softmax. Customer method (nsys `router_quant_topk_kernel` span, GPU 0,
1830 MHz, medians of 3 captures, `/raid/kimi/results/merge/cap_land`): FE 4.1 / 4.2 / 4.1 us MXFP4 and
4.0 / 4.1 / 3.9 us QOQ at M2 / M8 / M16, against 7.4-8.0 us for the swapab default captured the same
day (`cap_swapab`), i.e. -3.6..-3.9 us per FE launch; round-4 in-graph CUDA-event timing of the FE
alone gives 3.1-3.3 us (6.4 us without select-in-Mega; 3.6-3.8 us standalone back-to-back) and
FE+Mega gains ~2-3 us in every M x quant cell (E2E medians here: MXFP4 85.4 / 84.9 / 98.5, QOQ
94.1 / 85.1 / 93.5 us, within the host-skew spread of the swapab captures 85.1 / 85.5 / 96.4 and
76.2 / 108.5 / 98.6)
(`tests/test_select_in_mega.py`: 8 ranks, 50 seeds, 0 top-8 mismatches, y bit-identical). The
library default of `select_in_mega` stays 0 so standalone FE calls still produce
`topk_idx`/`topk_weights`. M2/M8/M16 FE values above are those medians (the value in parentheses: swapab default, same-day
captures); the wmma
value in parentheses is the previous default (96 x 4 WMMA, nsys span 8.3-8.7 us), the legacy value
the pre-tiny-M kernel. Round-4 finding: `DG_FE_ROUTER_L2_PERSIST=1` does not keep the router weights
resident across the Mega weight stream (no in-graph gain; the stamps-only win was a cold/warm-L2
artefact). E2E ranges span captures with different host launch skew.

QoQ correctness note (2026-09-11): `tests/test_four_api_correctness.py --frontend fe --tokens 32`
(real routing, 256 global tokens) failed for `qoq_mega_moe_fused` only (cos_min 0.0007, mxfp4 fine),
and so did the default balanced routing at qoq T=32 / T=64 (cos_min < 0). Root cause (fixed on this
branch): `promote_task_rf` of the QoQ inline-s2 RF loop (`kInlineS2`, `DG_FP4_QOQ_INLINE_S2`) read
accumulator `acc[..][j]` instead of `acc[..][i * 4 + j]`, so every token group `i > 0` of the RS
wgmma N dimension (tokens 8.. of a BM16 / BM24 block with > 8 valid rows, i.e. the T=32 / T=64 tiers;
BM8 tiers have a single group and were bit-exact) was promoted with token group 0's int32 sums under
its own activation scale. Localised with `--slot-check --hot-rows 12` at T=32: tokens 8..11 of the
hot expert matched the reference rows 0..3 (cos 0.9999) instead of their own. The interim host
gate (inline s2 only when `num_tokens x num_ranks <= 8`) is lifted; `DG_FP4_QIS2_MAX_GTOK` (default
0 = no bound) keeps it as a knob. Gates after the fix (8-rank, this branch): qoq T=2/8/8/16/32/64
cos_min 0.99993/0.99993/0.99993/0.99993/0.99993/0.99993 (T=128 is not a QoQ tier: host assert
`!qoq || plan.swap_ab`), `--frontend fe --tokens 32` qoq 0.99992 with a clean SLOT_CHECK, mxfp4
T=8/128/512 0.99997/0.99995/0.99990 (unchanged), 200-replay FE+Mega graph at M=16 qoq clean.
QoQ M16 with inline s2 back on (probe, rank 0): L1 phase 47.6 -> 43.3 us, L1 stage head-to-head
1424 -> 1265 ns; customer method (mega fused qoq M16, GPU0 median of last 3 spans): 77.2 us with the
gate vs 74.9 / 75.3 / 79.3 us over three captures with it lifted.

M2/M4 values are medians over five independent captures (branch tip with
`DG_FP4_STREAMK` default on); M8/M16 are the range over the r4/r5 captures
(`docs/` and `delivery/four_api_fable_timeline_last3_r4_20260909.md`,
`..._r5_20260909.md`).  Ranges reflect host launch skew between the eight
ranks, not kernel variance: the kernel duration on the latest-starting rank
agrees to within 1.5 us across captures.  Adding `dist.barrier()` before each
graph replay in the profiling driver (`DG_PROFILE_HOST_BARRIER=1`) removes
most of that skew (e.g. MXFP4 M8 110.8 -> 57.6 in one A/B).

### Round 5 (2026-09-12): FE lean entry point, customer method on 10.6.131.8 (8 x H20-3e, 1830 MHz, one build, one session)

`DG_FE_CC_LEAN=1` (now the default, docs/fe_cc_round5.md) compiles the cc router as its own 2.7 K-instruction kernel
(`router_cc_lean_kernel`) instead of the 11.9 K-instruction generic instantiation whose router role sat 73 KB into the
kernel and was fetched cold after every Mega weight stream (NCU top stall `no_instruction` 6.9 -> 1.7 warps per issue
cycle, instructions -44 %). Bit-identical x / x_sf / keys / topk (tests/fe5_ident.py, 256 cells byte-compared; 8-rank
gates in docs/fe_cc_round5.md section 4). `tests/fe5_campaign.sh`: 5 interleaved passes of base (`DG_FE_CC_LEAN=0`) and
lean, each with the FE's real routing ("normal") and with the FE output overridden by the Mega-only scope's balanced
assignment through one graph memcpy node (`DG_PROFILE_FORCE_BALANCED=1`, "balanced"); values = median over passes of
each capture's GPU-0 median-of-last-3 span (us); `skew` = inter-rank spread of the Mega kernel start (median over passes
of the per-pass max of the last 3 replays).

| Precision | M | routing | FE base | FE lean | E2E Mega base | E2E Mega lean | E2E base | E2E lean | Mega-only | skew base / lean |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | normal | 3.8 | **2.5** | 69.2 | 66.2 | 73.7 | 69.0 | 45.0 | 12.9 / 19.3 |
| MXFP4 | 2 | balanced | 4.1 | **2.6** | 49.4 | 60.0 | 54.8 | 64.2 | 45.0 | 43.9 / 62.1 |
| MXFP4 | 4 | normal | 4.1 | **2.5** | 69.5 | 90.2 | 73.4 | 93.0 | 54.3 | 13.8 / 50.8 |
| MXFP4 | 4 | balanced | 4.1 | **2.5** | 50.8 | 63.5 | 56.0 | 67.4 | 54.3 | 13.4 / 20.1 |
| MXFP4 | 8 | normal | 4.0 | **2.5** | 72.2 | 76.4 | 76.3 | 79.3 | 60.5 | 14.9 / 20.7 |
| MXFP4 | 8 | balanced | 4.1 | **2.5** | 76.8 | 64.1 | 82.2 | 68.0 | 60.5 | 24.2 / 16.8 |
| MXFP4 | 16 | normal | 4.0 | **2.8** | 89.4 | 92.2 | 93.5 | 95.4 | 78.7 | 20.9 / 13.7 |
| MXFP4 | 16 | balanced | 4.3 | **2.9** | 84.7 | 82.6 | 90.3 | 86.9 | 78.7 | 27.2 / 24.8 |
| QOQ | 2 | normal | 4.2 | **2.7** | 66.0 | 66.3 | 70.5 | 69.3 | 48.7 | 14.7 / 11.9 |
| QOQ | 2 | balanced | 4.3 | **2.8** | 47.3 | 43.9 | 52.7 | 48.2 | 48.7 | 14.5 / 14.7 |
| QOQ | 4 | normal | 4.3 | **2.7** | 72.0 | 69.2 | 76.6 | 72.2 | 50.8 | 23.3 / 17.6 |
| QOQ | 4 | balanced | 4.3 | **2.8** | 58.2 | 53.1 | 64.1 | 57.2 | 50.8 | 12.8 / 18.6 |
| QOQ | 8 | normal | 4.1 | **2.7** | 68.6 | 71.9 | 73.1 | 74.8 | 56.6 | 17.0 / 15.3 |
| QOQ | 8 | balanced | 4.3 | **2.8** | 66.6 | 56.0 | 72.5 | 60.1 | 56.6 | 20.2 / 23.0 |
| QOQ | 16 | normal | 4.4 | **2.8** | 91.4 | 95.3 | 95.7 | 98.5 | 78.5 | 14.3 / 17.5 |
| QOQ | 16 | balanced | 4.3 | **3.0** | 91.2 | 81.4 | 96.6 | 85.8 | 78.5 | 21.1 / 12.6 |

FE: -1.3 .. -1.6 us in all 32 cells, 5/5 passes each, base and lean pass distributions do not overlap (per-pass
values in docs/fe_cc_round5.md). E2E and E2E-Mega columns of the plain customer method are dominated by the
inter-rank launch skew (12-62 us), not by kernel work: the fused Mega kernels of the 8 ranks END within ~1 us of each
other (first in-kernel NVLink barrier), so GPU 0's Mega span = common end - GPU 0's own start, i.e. kernel work
(the latest rank's span, ~40 us at M2, = the Mega-only column) plus GPU 0's head start over the slowest rank
(10-35 us per replay). Per-rank decomposition of a balanced MXFP4 M2 replay (`scripts/decompose_e2e_skew.py`):
FE 4.0-4.6 us on every rank, FE end -> Mega start 1.2-1.5 us (1.1-1.2 us of it the forced-balanced memcpy node;
0.3 us with normal routing), Mega ends within 1.3 us across ranks, GPU 0 Mega span 55.5 = latest rank's 41.4 +
14.1 head start. Base-vs-lean differences in the E2E columns (+-5..20 us) are that skew's noise.

Streamed replays (separate method, do not mix with the table above: `DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30`, the
30 replays of a case enqueued back-to-back without per-iteration host sync / barrier, LEAN only, 3 passes + 1 Mega-only
pass, GPU-0 median of the last 3 replays then median over passes, us):

| Precision | M | routing | FE | E2E Mega | E2E | Mega-only (streamed) | Mega-start skew |
|---|---:|---|---:|---:|---:|---:|---:|
| MXFP4 | 2 | normal / balanced | 2.6 / 2.7 | 60.9 / 40.7 | 63.8 / 44.7 | 38.3 | 2.1 / 15.8 |
| MXFP4 | 4 | normal / balanced | 2.8 / 2.7 | 60.1 / 50.1 | 63.2 / 54.0 | 46.9 | 1.7 / 2.1 |
| MXFP4 | 8 | normal / balanced | 2.6 / 2.7 | 65.2 / 58.3 | 68.1 / 62.3 | 56.8 | 2.4 / 1.8 |
| MXFP4 | 16 | normal / balanced | 2.8 / 2.8 | 78.2 / 76.8 | 81.3 / 80.8 | 74.4 | 1.4 / 1.1 |
| QOQ | 2 | normal / balanced | 2.6 / 2.8 | 60.4 / 42.0 | 63.5 / 46.1 | 37.4 | 50.0 / 15.4 |
| QOQ | 4 | normal / balanced | 2.7 / 2.8 | 60.2 / 50.2 | 63.2 / 54.4 | 46.4 | 6.7 / 2.0 |
| QOQ | 8 | normal / balanced | 2.7 / 2.8 | 63.4 / 56.5 | 66.3 / 60.6 | 53.3 | 1.5 / 1.4 |
| QOQ | 16 | normal / balanced | 3.0 / 3.0 | 76.6 / 76.0 | 80.0 / 80.5 | 73.0 | 1.6 / 1.5 |

Streamed, the ranks lock to 1-2.5 us of Mega-start skew in 12 of 16 cells (M2 and qoq M4 normal stay looser: one
token per TP half gives the collectives little to pin on), the balanced E2E reads FE + 1.3 us memcpy node + Mega
(Mega-only + 1-3 us), and the normal-routing Mega (60-65 us at M2-8) versus balanced (40-58) is the real cost of the
FE's actual unbalanced top-8 routing, not skew.

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
