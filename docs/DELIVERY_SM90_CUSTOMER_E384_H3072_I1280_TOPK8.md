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

Performance policy: report the arithmetic mean across all eight ranks. Correctness uses the real router path, then compares the TP ReduceScatter pipeline against the AllReduce/reference-token pipeline. The deterministic balanced route is applied after the real router frontend so performance has exactly eight assignments per EP rank.

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
