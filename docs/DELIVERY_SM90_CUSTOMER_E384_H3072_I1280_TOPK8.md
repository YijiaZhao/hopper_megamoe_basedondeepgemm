# Delivery: SM90 Customer MegaMoE E384/H3072/I1280/TopK8

Frozen source branch: `w_int4_a_int8_qserve_and_w_mxfp4_a_fp8`
Base commit: `a8ff0dbc3acddd0f68b63206bb74e3f3170e5e8c`

This delivery contains the customer TP4/DP2/EP8 scripts and the preserved local
implementation changes that produced the August 27, 2026 result set.

## Fixed shape

- 8 x H20-3e, TP4 / DP2 / EP8
- E=384, local E=48, H=3072, I=1280, TopK=8
- BS=1, ISL=4 per DP replica; M_moe=4 per TP group
- After TP4 ReduceScatter: one token per physical rank
- Balanced performance routing: eight assignments per EP rank

## Entry points

- Correctness: `tests/test_customer_bs1_isl4_tp4_dp2_correctness.py`
- Full E2E: `tests/bench_customer_bs2_tp4_dp2.py`
- Core: `tests/bench_customer_core_balanced.py`
- Communication: `tests/bench_customer_comm.py`

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

## Historical validated results

- scaled-MXFP4: exact=1, max_abs=0, cos_min=0.9999999404;
  core max-rank 131.067 us; E2E max-rank 276.357 us.
- canonical QoQ W4A8: exact=1, max_abs=0, cos_min=0.9999999404;
  core max-rank 121.118 us; E2E max-rank 288.904 us.

See `HANDOFF_SM90_CUSTOMER_NEW_SHAPE_20260827.md` for the exact historical
commands and timing scope.

## Important

Do not use `/raid/kimi/customer_megamoe_local_a8ff` as a source of truth after
August 27; it was later modified by unified/Green experiments. This Git commit
is the frozen delivery source. Do not mix APIs or layout files from GitHub
`main@7124ab8` or later experimental worktrees.

## Recovered runnable frontend and current mean-rank results

The customer-shape scaled-MXFP4 frontend is implemented as a two-kernel fallback:

```text
SM90 BF16 router GEMM
-> mxfp4_quant_topk_from_logits_kernel (FP8 E4M3, K128 scale, TopK8)
-> production SM90 MegaMoE
```

Relevant symbols/files:

- `csrc/mega_frontend.cu`: `mxfp4_quant_topk_from_logits_kernel`
- `csrc/mega_frontend.h`: `launch_mxfp4_quant_topk_from_logits`
- `csrc/apis/mega.hpp`: `mxfp4_router_quant_topk_split`

Performance policy: report the arithmetic mean across all eight ranks. Correctness uses the normal, unmodified real-router output. Performance uses the real frontend where applicable and then replaces its routing metadata with the deterministic balanced pattern, giving exactly eight assignments per EP rank.

Fresh H20-GPU-08 results on 2026-08-31, clocks requested at 1980 MHz:

| precision | real-router correctness | MegaMoE core rank mean | full E2E rank mean |
|---|---|---:|---:|
| scaled-MXFP4 | exact=1, max_abs=0, cos_min=0.9999999404 | 155.583 us | 276.145 us |
| QoQ W4A8 | exact=1, max_abs=0, cos_min=0.9999999404 | 146.868 us | 289.724 us |

The core benchmark uses Kineto's sum of the matching SM90 MegaMoE kernels and reports the mean over eight ranks. Full E2E uses CUDA events around 150 iterations and reports the mean of the eight per-rank averages.

### Reproduction

```bash
ssh root@10.6.131.8
nvidia-smi -pm 1
nvidia-smi -lgc 1980

docker exec ds_new bash -lc '
  cd /raid/kimi/a8ff_exact_local_20260831
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export DG_SM90_MOE_GREEN_OVERLAP=0

  # scaled-MXFP4
  export DG_W4A8_INT=0 DG_MXFP4_REL_LUT=1 DG_MXFP4_ABS_SCALE256=1
  python3 tests/test_customer_bs1_isl4_tp4_dp2_correctness.py
  python3 tests/bench_customer_core_balanced.py \
    --num-processes 8 --attn-tp-size 4 --batches 1 \
    --hidden 3072 --intermediate-hidden 1280 --num-experts 384 \
    --num-topk 8 --block-n 128 --warmup 10 --num-tests 50
  python3 tests/bench_customer_bs2_tp4_dp2.py \
    --tp-size 4 --batches 4 --balanced-routing \
    --valid-tokens-per-group 4 --warmup 15 --iters 150

  # QoQ INT4 x INT8
  export DG_W4A8_INT=1
  unset DG_MXFP4_REL_LUT DG_MXFP4_ABS_SCALE256
  python3 tests/test_customer_bs1_isl4_tp4_dp2_correctness.py
  python3 tests/bench_customer_core_balanced.py \
    --num-processes 8 --attn-tp-size 4 --batches 1 \
    --hidden 3072 --intermediate-hidden 1280 --num-experts 384 \
    --num-topk 8 --block-n 128 --warmup 10 --num-tests 50
  python3 tests/bench_customer_bs2_tp4_dp2.py \
    --tp-size 4 --batches 4 --balanced-routing \
    --valid-tokens-per-group 4 --warmup 15 --iters 150
'
```

## Clean-main verification — 2026-08-31

A clean copy of local `main@25681fd` was transferred to H20-GPU-08, rebuilt from source in the `ds_new` container, and tested independently of earlier build artifacts.

Configuration:

```text
8 x H20-3e, clocks requested at 1980 MHz
TP4 / DP2 / EP8
E=384, H=3072, I=1280, TopK=8
BS=1, ISL=4 per DP replica
Performance routing: balanced, 8 assignments per EP rank
Correctness routing: normal real Router GEMM + quantization + TopK8 output (not overwritten)
```

Correctness:

| precision | exact | max_abs | cos_min |
|---|---:|---:|---:|
| scaled-MXFP4 | 1 | 0 | 0.9999998808 |
| QoQ INT4 x INT8 | 1 | 0 | 0.9999998212 |

Full E2E, arithmetic mean across eight ranks (`warmup=15`, `iters=150`):

| precision | mean-rank E2E |
|---|---:|
| scaled-MXFP4 | 276.089 us |
| QoQ INT4 x INT8 | 281.006 us |

MegaMoE kernel-only measurements were repeated three times with `warmup=20` and `num_tests=200`. Each value below is the arithmetic mean across eight ranks:

| precision | repeat 1 | repeat 2 | repeat 3 | mean of repeats |
|---|---:|---:|---:|---:|
| scaled-MXFP4 | 108.984 us | 125.195 us | 113.564 us | 115.914 us |
| QoQ INT4 x INT8 | 140.911 us | 107.317 us | 115.829 us | 121.352 us |

The kernel-only numbers show noticeable run-to-run variation; retain all repeats rather than quoting only the best run. The E2E results above are the primary production comparison.

Verification logs on H20-GPU-08:

```text
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_build.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_mxfp4_correctness.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_mxfp4_core.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_mxfp4_core_rep1.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_mxfp4_core_rep2.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_mxfp4_core_rep3.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_mxfp4_e2e.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_int4_correctness.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_int4_core.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_int4_core_rep1.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_int4_core_rep2.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_int4_core_rep3.log
/raid/kimi/megamoe_opt2_results/verify_main_25681fd_int4_e2e.log
```
