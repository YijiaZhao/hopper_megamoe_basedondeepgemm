# Handoff: SM90 Customer MegaMoE — New Shape / TP4-DP2-EP8

Date: 2026-08-27

## 1. Only objective after `/clear`

Develop and benchmark the customer Hopper/H20 MegaMoE for the **fixed new production case** below. Compare:

1. **PR #323 style:** SM90 L1/L2 unified persistent kernel.
2. **PR #357 style:** Green Context split pipeline (`K1 -> K2 -> K3`).

Select the faster correct implementation. Do not resume old TP2/TP4 tables or other batch sizes.

## 2. Fixed customer configuration

```text
GPU/node: H20-3e x 8
World: 8
Attention TP: 4
Attention DP: 2
Expert Parallel: 8
Experts: 384
Hidden: 3072
Intermediate: 1280
TopK: 8
Input per DP replica: BS=1, ISL=4 => 4 unique tokens / TP group
Global unique tokens before TopK: 4 * DP2 = 8
After TP4 ReduceScatter: 1 real token / physical rank
Total routed assignments: 8 tokens * TopK8 = 64
Forced balanced routing: exactly 8 assignments / EP rank
Precision modes: scaled-MXFP4 and canonical QoQ W4A8
```

No token or intermediate padding is needed:

```text
M_per_TP_group = 4, divisible by TP4
I=1280, divisible by 128
H=3072, divisible by 128
```

Balanced routing used by the custom scripts routes every source token once to every EP rank. Each EP rank owns 48 experts and receives exactly 8 assignments.

## 3. Current measured baseline

H20 clocks were requested at 1980 MHz:

```bash
ssh root@10.6.131.8
nvidia-smi -pm 1
nvidia-smi -lgc 1980
```

Full timing scope:

```text
TP4 ReduceScatter
-> Router GEMM
-> Quant + TopK8
-> MegaMoE dispatch + L1 + SwiGLU/quant + L2 + EP combine
-> TP AllGather
```

Stable max-rank results (`warmup=15`, `iters=150`):

| Mode | Full E2E |
|---|---:|
| scaled-MXFP4 | **276.357 us** |
| QoQ W4A8 | **288.904 us** |

Component measurements:

| Component | scaled-MXFP4 | QoQ |
|---|---:|---:|
| MegaMoE core, balanced, local M=1/rank, rank max | 131.067 us | 121.118 us |
| TP4 RS + AG, H3072, max rank | 72.544 us | 72.544 us |

The current new-shape frontend is a **split fallback**:

```text
BF16 Router GEMM kernel
-> Quant + TopK kernel
```

It is not yet the old shape-specific fused WGMMA frontend. A first generic scalar fused frontend was ~1.7 ms and was abandoned.

## 4. Correctness status

Custom TP correctness gate compares:

```text
RS -> frontend -> forced-balanced metadata -> MegaMoE -> AG
vs
AR -> identical contiguous token slice -> same frontend/metadata/MegaMoE -> AG
```

Results:

```text
scaled-MXFP4: exact=1, max_abs=0, cos_min=0.9999999404
QoQ:          exact=1, max_abs=0, cos_min=0.9999999404
```

This validates TP/DP/EP pipeline equivalence for the fixed new case. Independent dequant arithmetic tests remain in the original repo.

## 5. Working locations

### Main active H20 workspace

```text
/raid/kimi/customer_megamoe_local_a8ff
```

This was synced from the Mac repo:

```text
/Users/kimiz/Downloads/agent/cc/hopper_megamoe_basedondeepgemm
branch: w_int4_a_int8_qserve_and_w_mxfp4_a_fp8
base commit: a8ff0dbc
```

Do not overwrite the Mac working tree; it has uncommitted changes.

### Container used for build/run

```text
container: ds_new
PyTorch: 2.11.0+cu130
CUDA toolkit: 13.0
```

Build command:

```bash
docker exec ds_new bash -lc '
  cd /raid/kimi/customer_megamoe_local_a8ff &&
  export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH &&
  bash develop.sh
'
```

### Official PR references

```text
/raid/kimi/DeepGEMM_pr357
commit: bb837421 (PR #357 head)
```

Public clone used for git history:

```text
/raid/kimi/deepgemm_w4a8_mega_moe_on_hopper
```

Avoid using its current `main` build as the active implementation; public main `7124ab8` was internally inconsistent when built against the current source layout.

## 6. Custom scripts

### Full E2E performance

```text
/raid/kimi/customer_megamoe_local_a8ff/tests/bench_customer_bs2_tp4_dp2.py
```

Despite the historical filename, it now represents the final case:

```text
BS1 x ISL4 per DP, global 8 tokens, TopK8
```

Run MXFP4:

```bash
docker exec \
  -e DG_MXFP4_REL_LUT=1 \
  -e DG_MXFP4_ABS_SCALE256=1 \
  ds_new bash -lc '
    cd /raid/kimi/customer_megamoe_local_a8ff &&
    python tests/bench_customer_bs2_tp4_dp2.py \
      --tp-size 4 --batches 4 --balanced-routing \
      --valid-tokens-per-group 4 --warmup 15 --iters 150
  '
```

Run QoQ:

```bash
docker exec -e DG_W4A8_INT=1 ds_new bash -lc '
  cd /raid/kimi/customer_megamoe_local_a8ff &&
  python tests/bench_customer_bs2_tp4_dp2.py \
    --tp-size 4 --batches 4 --balanced-routing \
    --valid-tokens-per-group 4 --warmup 15 --iters 150
'
```

### Correctness

```text
tests/test_customer_bs1_isl4_tp4_dp2_correctness.py
```

MXFP4:

```bash
docker exec \
  -e DG_MXFP4_REL_LUT=1 \
  -e DG_MXFP4_ABS_SCALE256=1 \
  ds_new bash -lc '
    cd /raid/kimi/customer_megamoe_local_a8ff &&
    python tests/test_customer_bs1_isl4_tp4_dp2_correctness.py
  '
```

QoQ:

```bash
docker exec -e DG_W4A8_INT=1 ds_new bash -lc '
  cd /raid/kimi/customer_megamoe_local_a8ff &&
  python tests/test_customer_bs1_isl4_tp4_dp2_correctness.py
'
```

### Core-only balanced benchmark

```text
tests/bench_customer_core_balanced.py
```

### Communication-only

```text
tests/bench_customer_comm.py
```

## 7. New-shape frontend changes already made

Files changed in the active H20 workspace:

```text
csrc/mega_frontend.cu
csrc/mega_frontend.h
csrc/apis/mega.hpp
```

Added support for:

```text
H3072 / E384 / TopK8
```

Current performant fallback is:

```text
sm90_bf16_gemm router
-> generic MXFP4 or QoQ quant/topk kernel
```

The original fused frontend remains selected for the old exact shape:

```text
H4096 / E128 / TopK6
```

## 8. PR #323 — unified SM90 direction

Official DeepGEMM PR #323 is merged and contains an SM90 all-FP8 MegaMoE in one persistent kernel.

Relevant structure:

```text
dispatch -> L1 -> SwiGLU/quant -> L2 -> combine
```

The scheduler switches between:

```text
BlockPhase::Linear1
BlockPhase::Linear2
```

For our MXFP4 history, the commit immediately before the local split was introduced is:

```text
5e012d4 Optimize SM90 MegaMoE kernels
615ba0a Split SM90 MegaMoE into L1 and L2 kernels
```

Fastest coarse unified comparison path:

```bash
cd /raid/kimi/deepgemm_w4a8_mega_moe_on_hopper
git worktree add /raid/kimi/customer_megamoe_unified_5e012d4 5e012d4
```

Then forward-port only:

- E384/H3072/I1280/TopK8 shape support
- current MXFP4 scaled exponent path
- current QoQ SHIFTXOR path if possible
- fixed balanced benchmark scripts

This should be done before a full Green Context port because it gives the quickest same-precision latency comparison.

## 9. PR #357 — Green Context split direction

PR #357 is an open Draft for **SM100/CUDA 13.1**, not SM90. It is nevertheless the correct architecture reference.

It splits MegaMoE into:

```text
K1: dispatch_l1_swiglu
K2: l2_combine
K3: combine_reduce
```

K1 and K2 run concurrently in separate Green Context execution contexts. K3 runs after both.

Critical difference from the failed local prototype: PR #357 does **not** reuse the fused MegaMoE workspace/barrier protocol. It introduces:

```text
SplitWorkspace
SplitStateOffset::K1ReadyTasks
SplitStateOffset::K1DoneBlocks
SplitStateOffset::K2ClaimCounter
SplitStateOffset::K2DoneTasks
SplitStateOffset::K2DoneBlocks
SplitStateOffset::K3DoneElements
```

It also uses multiple buffers and graph replay. Test comments say the split path is usually slightly faster than the fused kernel on SM100, but this must not be assumed for H20 tiny-M.

### H20 Green Context facts already verified

On H20-GPU-08:

```text
Total SMs: 78
minSmPartitionSize: 8
smCoscheduledAlignment: 8
```

With `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`, exact split works:

```text
52 SM + 26 SM
```

A live probe successfully launched a kernel on a Green stream and accessed `cudaMalloc` memory.

Current CUDA 13.0 environment supports Driver API Green Context:

```text
cuDeviceGetDevResource
cuDevSmResourceSplitByCount
cuGreenCtxCreate
cuGreenCtxStreamCreate
```

PR #357's execution-context CUDA Graph API requires CUDA 13.1. For the SM90 port either:

1. use Driver Green streams/events on CUDA 13.0; or
2. move to a CUDA 13.1 container and port the graph design directly.

## 10. Failed experimental Green patch — do not treat as finished

An experimental direct-concurrency patch was added to the active H20 workspace:

- `dispatch_producer_sms` separated from L2 kernel `kNumSMs`
- `LaunchRuntime::launch_on_stream`
- Driver Green contexts 52/26
- L2 launched first, L1 second

It compiles, but the 8-GPU correctness test fails with:

```text
CUDA error: unspecified launch failure
```

Cause: the original sequential L1/L2 kernels still reuse grid-sync and NVLink barrier slots. Directly running them concurrently corrupts barrier state.

Do not continue by adding more ad-hoc tags only. Replace this with PR #357's dedicated split workspace/state protocol, or disable the experimental path while testing unified mode.

Environment variable for the failed experimental path:

```text
DG_SM90_MOE_GREEN_OVERLAP=1
```

Default is off.

## 11. Recommended next execution order

### Step A — unified kernel coarse test first

1. Create worktree at `5e012d4`.
2. Build in `ds_new`.
3. Adapt exact shape and TopK8.
4. Run core-only and full E2E MXFP4.
5. If promising, port QoQ decode.

### Step B — proper Green split port

Port from PR #357, not from the failed direct-concurrency patch:

1. Port `SplitWorkspace` and `SplitStateOffset`.
2. K1 = existing SM90 dispatch + L1 + SwiGLU/quant.
3. K2 = existing SM90 L2 + combine-scatter, claiming only ready tasks.
4. K3 = BF16 top-k combine-reduce.
5. Use at least two state/symmetric buffers for pipelining.
6. Test SM partitions:
   ```text
   48/30
   52/26
   56/22
   ```
7. Correctness must remain exact against the sequential path.
8. Report both isolated latency and steady-state replay throughput.

### Step C — select final strategy

Use max-rank full E2E for the fixed customer case. Do not select based on core-only or average-rank timing.

## 12. Kernel Factory status

Kernel Factory authentication is expired:

```text
token expired and refresh failed
```

User must run on the Mac:

```bash
kf auth login
```

After login, Kernel Factory can be used for SM90 K1/K2 kernel tuning, but host-level multi-GPU Green Context orchestration still needs the local harness/environment.

## 13. Acceptance criteria

Do not call the task complete until all are true:

1. At least one optimized path beats the current full E2E baseline.
2. MXFP4 and QoQ correctness pass.
3. Results include max-rank wall time.
4. Timing includes RS + frontend + MegaMoE + AG.
5. The exact customer token/topology convention is documented.
6. Code and run commands are reproducible.
