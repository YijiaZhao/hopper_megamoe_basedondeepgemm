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
