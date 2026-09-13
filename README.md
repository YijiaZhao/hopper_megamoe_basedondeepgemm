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
bash tests/fe5_gates.sh                                     # the gate set above, T = 1 2 8 16 32 + slot checks
OUT=/raid/kimi/results/fe5c bash tests/fe5c_sweep.sh ; python3 scripts/summarize_fe5c_sweep.py /raid/kimi/results/fe5c/*.log
python3 tests/fe_dump_compare.py --root-a <reference checkout> --root-b . --seeds 64 --ref-torch   # 1 GPU
```

## Performance results (2026-09-12)

Microseconds; M = global tokens over the 8 ranks (M2 / M4 / M8 = 1 row per rank, M16 = 2 rows);
fused backend (`mxfp4_mega_moe_fused` / `qoq_mega_moe_fused`); GPU 0, median of the last 3
replays, median over 3 passes.

| | Precision | M2 | M4 | M8 | M16 |
|---|---|---:|---:|---:|---:|
| E2E, normal routing | MXFP4 | 63.8 | 63.2 | 68.1 | 81.3 |
| E2E, normal routing | QoQ | 63.5 | 63.2 | 66.3 | 80.0 |
| E2E, forced-balanced | MXFP4 | 44.7 | 54.0 | 62.3 | 80.8 |
| E2E, forced-balanced | QoQ | 46.1 | 54.4 | 60.6 | 80.5 |
| Mega-only, balanced | MXFP4 | 38.3 | 46.9 | 56.8 | 74.4 |
| Mega-only, balanced | QoQ | 37.4 | 46.4 | 53.3 | 73.0 |
| FE kernel | both | 2.6–2.8 | 2.6–2.8 | 2.6–2.8 | 2.8–3.0 |

Rows: **E2E** = frontend kernel + fused MegaMoE kernel in one CUDA graph, span from the frontend
start to the MegaMoE end; **normal routing** = the frontend's real top-8 of random hidden rows;
**forced-balanced** = the frontend runs, then one graph memcpy node overrides its routing with
the balanced assignment (`expert = s * 48 + (g + 7 s) % 48`, weight 1/8; `DG_PROFILE_FORCE_BALANCED=1`);
**Mega-only** = the fused MegaMoE kernel alone with the same balanced assignment; **FE kernel** =
the frontend kernel span (`router_cc_lean_kernel`, `DG_FE_SELECT_IN_MEGA=1`: the kernel ends after
the keys, the MegaMoE prologue selects). At M2 / M4 only the ranks that own a token carry a real row
(ranks 0, 4 / 0, 1, 4, 5); the other ranks' row is all-zero padding, which the frontend leaves
unrouted (`DG_FE_ZERO_ROW_UNROUTED=1`, exact: x = 0 gives y = 0). The 256 MiB L2 flush, the TP4
reduce-scatter and the TP4 all-gather sit between the replays, outside the graph.

**Measurement method:** host `10.6.131.8`, eight H20-3e with the SM clock locked at 1830 MHz
(`nvidia-smi -lgc 1830,1830`, verified by the capture script), container `fe5c_build` (the
pinned image above), kernels of `perf/phase-stamps-probe` `eba23b1` (build `fee8931`; the default
code paths of this tree), one session 2026-09-12 08:05-08:21 UTC.
`scripts/capture_four_api_h20_timelines.sh` runs `tests/profile_four_api_h20.py` under Nsight
Systems (`--trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none`) with
`DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30`: after 2 warm-up replays, the 30 measured replays of a
case are enqueued back-to-back on the stream (L2 flush, input copy, reduce-scatter, graph replay,
all-gather; no per-iteration `torch.cuda.synchronize()` / `dist.barrier()`), so the ranks align
through the on-stream collectives. `scripts/summarize_four_api_h20_last3.py` reads the report of
GPU 0, takes the last 3 replays, measures each span (E2E: frontend kernel start -> MegaMoE kernel
end; Mega-only: kernel start -> end; FE: frontend kernel start -> end) and reports their median;
`tests/fe5_summarize_campaign.py` takes the median over the 3 passes (Mega-only: 1 pass). Env of
the runs: `DG_FE_SELECT_IN_MEGA=1 DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30`, plus
`DG_PROFILE_FORCE_BALANCED=1` for the forced-balanced rows; every other knob at its library default.

Reproduce (inside the container, repo root; 8 idle GPUs, clock locked):

```bash
# E2E normal + forced-balanced, 3 passes each, then one Mega-only pass; prints the per-capture tables
bash tests/fe5_campaign_streamed.sh 3 /raid/kimi/results/fe5/cap_streamed            # ~16 min
python3 tests/fe5_summarize_campaign.py /raid/kimi/results/fe5/cap_streamed          # medians over the passes
# one capture set by hand (what the campaign script calls)
DG_FE_SELECT_IN_MEGA=1 DG_PROFILE_STREAMED=1 DG_PROFILE_ITERS=30 DG_PROFILE_FORCE_BALANCED=0 \
  OUT=/raid/kimi/results/x SCOPES=e2e BACKENDS=fused QUANTS="mxfp4 qoq" TOKENS_LIST="2 4 8 16" FORCE=1 \
  bash scripts/capture_four_api_h20_timelines.sh
SCOPES=e2e BACKENDS=fused python3 scripts/summarize_four_api_h20_last3.py /raid/kimi/results/x
```

Env knobs of this tree (all read once per process; the table above uses the defaults except where noted):

| Knob | Default | Meaning |
|---|---|---|
| `DG_FE_SELECT_IN_MEGA` | 0 (the profiling driver and the campaign scripts set 1) | cc frontend ends after the 384 keys per token; the fused MegaMoE prologue selects top-8 + softmax |
| `DG_FE_ROUTER_L2_PERSIST` | 1 on the cc path, 0 otherwise | router weights in the persisting-L2 set-aside (access-policy-window launch attribute) |
| `DG_FE_ROUTER_WLAYOUT` | `fragment` | swapab path: router weights permuted once into m16n8k16 A-fragment order (`fable_router_weight_fragment_layout`); `row`, `pre` (caller permuted) |
| `DG_FE_STAMPS` | 0 | per-CTA `%globaltimer` phase stamps of the frontend kernel (`fable_frontend_stamps`) |
| `DG_FE_FORCE_BALANCED` | 0 | `tests/test_four_api_correctness.py`: the forced-balanced override, reference follows it |
| `DG_FE_ZERO_ROW_UNROUTED` | 1 | cc frontend: an all-zero hidden row (a padding row of a rank that owns no token) is left unrouted (all-zero keys / `topk_idx` -1, weight 0) instead of tie-broken onto experts 0..7 on rank 0; exact (x = 0 gives y = 0 either way), non-zero rows bit-identical |
| `DG_FP4_FINE_COMBINE` | 1 | per-token arrival counters replace the combine NVLink barrier (combine warps claim tokens from a per-launch ticket) |
| `DG_FP4_PUSH_DISPATCH`, `DG_FP4_PUSH_DISPATCH_MAX_M` | 1, 16 | source rank pushes routed rows + tickets into the destination pool during routing (<= MAX_M global tokens) |
| `DG_FP4_LEAN_ROUTING` | 1 | publish non-zero experts only; no cross-rank count broadcast under push |
| `DG_FP4_PUSH_DONE_FLAGS` | 1 | per-rank DONE count replaces NVLink barrier #1 (lean push) |
| `DG_FP4_QOQ_INLINE_S2` | 1 | QoQ: fold s2 into the int8 weight at decode, one int32 set per task, one promote |
| `DG_FP4_QIS2_PREFETCH_PACKED` | 1 | QoQ inline-s2 loop: packed-word LDS before the wgmma wait |
| `DG_FP4_STREAMK`, `DG_FP4_STREAMK_MAX_M` | 1, 8 | stream-K unit ranges over all 78 SMs when the launch has fewer L1 tasks than SMs (M <= 4 in practice) |
| `DG_FP4_L1_BN`, `DG_FP4_L2_BN`, `DG_FP4_BN512_MIN_M`, `DG_FP4_BN512_MAX_M` | 512, 256, 16, 16 | wide (512-row) L1 tasks for M = 16 |
| `DG_FP4_SPLITK_L1` | 1 | L1 tasks of the last partial wave run as two K halves |
| `DG_FP4_DIST_BCAST` | 1 | dispatch expert-count broadcast spread over all SMs |
| `DG_FP4_PREFETCH_KBLOCKS` | 0 under push dispatch or >= 16 rows per rank, else 8 | weight K-blocks prefetched into L2 per task while waiting for activations |
| `DG_FP4_BM8_STAGES` | 4 | pipeline depth of the BM8 tier (2 K128 blocks per stage; 1 with wide tasks) |
| `DG_FP4_HALF_TILE`, `DG_FP4_L2_HALFROW`, `DG_FP4_SPLITK_L2`, `DG_FP4_NVL_FAST_EPI`, `DG_FP4_SWAP_PIPE`, `DG_FP4_QIS2_FRAGS`, `DG_FP4_QIS2_ILV`, `DG_FP4_RF_PREFETCH_PACKED`, `DG_FP4_POOL_STRIDE_DEBUG`, `DG_FP4_SPIN_TIMEOUT` | 0 / 2 (frags) | alternative task shapes and debug switches, off in the table above (documented in `csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp`) |
| `DG_PROFILE_STREAMED`, `DG_PROFILE_ITERS` | 0, 8 | profiling: back-to-back replays (1) and replay count; the table uses 1 / 30 |
| `DG_PROFILE_FORCE_BALANCED` | 0 | profiling: forced-balanced routing in the E2E scope (memcpy override) |
| `DG_PROFILE_HOST_BARRIER` | 0 | profiling: `dist.barrier()` before each replay when not streamed |
| `DG_BENCH_FLUSH_L2_BYTES` | 256 MiB | profiling: L2 flush size between replays |

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
