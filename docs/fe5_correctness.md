# Correctness verification: torch-MoE reference, frontend layout gate, forced-balanced routing, sweep

Machines: 10.6.131.8 (8 x H20, container `fe5c_build` = the pinned `lmsysorg/sglang@sha256:687efca0...` image) for the
one-seed matrix and the gates; ComputeLab job 4252043 (viking-prod-583, 8 x H20-3e, `lmsysorg/sglang:dev`) for the 50-seed
sweep. Kernels: the default code paths of this tree (frontend `router_cc_lean_kernel` for <= 2 rows per rank, swapab above;
fused MegaMoE with the library-default knobs; `DG_FE_SELECT_IN_MEGA` 1 = the pipeline configuration, 0 = the frontend selects).

## 1. The reference: a real-routing pure-torch MoE (`--reference torch-moe`)

`tests/test_four_api_correctness.py` reference flavours:

| flag | routing the reference uses | activations | contains our kernels? |
|---|---|---|---|
| default (`--frontend none`) | synthetic balanced (`slot * 48 + (8 g + 7 slot) % 48`, weight 1/8) written into the buffer | torch per-token cast | no, but routing is not real |
| `--frontend fe` | the frontend kernel's own topk_idx / topk_weights read back from the buffer | the frontend kernel's x / x_sf | the reference depends on the frontend outputs |
| `--frontend fe --router-ref torch` == `--reference torch-moe` | pure torch: bf16 router GEMM (fp32 accumulate) -> bf16-rounded logits -> top-8 (value desc, index asc on ties) -> fp32 softmax over the 8 selected bf16 logits (`expf(v - max) / sum`) | torch per-token cast (`per_token_cast_to_fp8` / `per_token_cast_to_int8`) | no |

The MoE part is the same in every flavour: for every (global token, slot) owned by this rank, the reference computes
`L1 = dequant(W1[e]) @ x_ref`, clamps gate / up (`activation_clamp`), `mid = silu(gate) * up * w`, quantises `mid` to the
kernel's intermediate format (fp8 per-64 group amax/448 with power-of-two scale for MXFP4, int8 per-64 amax/127 for QoQ),
`dequant(W2[e]) @ mid` rounded to bf16, then sums the 8 routes across the 8 EP ranks (all-reduce) and rounds to bf16.
Weight dequantisation is the kernel-exact one: MXFP4 `dequantize_mxfp4_kernel_exact_fp32(e2m1, e8m0, row reference exponent)`,
QoQ `dequantize_qoq_to_fp32(int4, s1, s2)` (deep_gemm/quantization_*_fused.py). `--reference torch-moe` therefore checks the
kernels against the same weights with real router routing and none of our kernels in the reference.

Reporting: the `ROUTER_REF` line gives, per launch, top-8 index-set agreement (tokens), weight max |diff| on agreeing tokens
and over all tokens, x byte diffs (bytes, rows, max |dq| in quantisation steps) and x_sf word diffs; disagreeing tokens are
printed with the experts that differ (`ROUTER_REF_DISAGREE`). With `DG_FE_SELECT_IN_MEGA=1` the test decodes the top-8 /
softmax from the compact key array in python (`_topk_from_keys`), reads the routing the Mega prologue wrote into the buffer
after the kernel and reports `SELECT_IN_MEGA ... N tokens differ`. `--global-tokens 2 | 4` (owner layout, M < 8) runs the
frontend on every rank and unroutes the inactive ranks' rows (topk_idx -1 / all-zero keys, as `DG_PROFILE_FORCE_BALANCED`
does). `--seeds N` runs N seeds in one process (same expert / router weights, new hidden rows) and prints a `SUMMARY` line.

Pass criteria: finite, `cos_min >= 0.99`, `0.97 <= norm ratio <= 1.03`; `--slot-check`: every rank's per-(token, slot) L2
partials have cos >= 0.999 against the reference route.

## 2. Frontend output layout gate: `tests/fe_dump_compare.py`

Runs the frontend alone (single GPU, `types.SimpleNamespace` buffer, 64 rows) for `--seeds` x rows {1, 2} x {mxfp4, qoq}
under two configurations (two env settings and / or two builds via `--root-a` / `--root-b`; one child process each) and
byte-compares every buffer the fused Mega consumes, after pre-filling the buffers with a sentinel so unwritten slots are visible:

| buffer | layout | semantics |
|---|---|---|
| `x` | [4, 3072] bytes (rows 0..3 = incl. the padding rows the tiny-M Mega reads) | mode 0 (mxfp4): fp8 e4m3 per K128 group; mode 1 (qoq): int8 whole row (stored in the e4m3 tensor) |
| `x_sf` | [4, 24] fp32 | mode 0: per-K128 amax / 448; mode 1: whole-row amax / 127 replicated into the 24 slots |
| `topk_idx`, `topk_weights` | [4, 8] int64 / fp32 | select_in_mega=0 launch |
| `ticket` | workspace [0, 256) | tickets / hand-off counters |
| `x_sel1`, `x_sf_sel1` | as x / x_sf | written by the select_in_mega=1 launch |
| `keys` | workspace [65792, 65792 + 4096) | compact [token][384] u32 keys (orderable bf16 << 16 \| 0xFFFF - expert), token t at + t * 1536 |
| `ticket_sel1` | workspace [0, 256) | after the select_in_mega=1 launch |

Per buffer it prints mismatch count, first mismatching flat index (row / col), both values (raw byte + decoded e4m3 / int8 /
fp32 / key -> expert) and a class: `identical`, `zeros/uninitialised`, `whole-row missing`, `rounding` (<= 1 quantisation
step / 2^-20 relative), `layout shift`, `other`. `--ref-torch` additionally compares each dump's x / x_sf with the torch
per-token casts. `DG_FE_SELECT_IN_MEGA` 0 vs 1 is always covered (x / x_sf of the two launches are compared inside each dump).

Result (64 seeds, `--ref-torch`): `FE_DUMP_COMPARE cells=256 failing_cells=0`, all 9 buffers identical in all 256 cells,
padding rows included; mxfp4 x / x_sf byte-identical to the torch per-token casts in every cell; qoq x differs from
`per_token_cast_to_int8` by exactly one quantisation step on a handful of elements (12 of 64 cells per rows value, <= 56
bytes), x_sf identical: an int8 rounding-tie difference (kernel `v * (1/scale)` + round vs torch `x / scale` + round), not a
layout difference, and inside the reference's tolerance (it is what the `qoq x bytes diff` column of the sweep shows).

## 3. Forced-balanced routing

The "forced-balanced" E2E rows of the README come from `DG_PROFILE_FORCE_BALANCED=1` (tests/profile_four_api_h20.py;
`tests/fe5_campaign_streamed.sh`, captures `lean_balanced_p*`). E2E scope only: the frontend kernel runs exactly as in the
normal mode; one graph memcpy node (`cudaMemcpyAsync`, no kernel; inside the E2E span, outside both kernel spans) then
overwrites the frontend's routing output before the Mega:
* assignment (`forced_balanced_routing`): for active row t of rank r (global token g = r * local_rows + t), slot s ->
  expert `s * 48 + (g + 7 s) % 48`, i.e. one route to every EP rank, weight 1/8 each; inactive owner-layout rows (M < 8) are
  unrouted;
* with `DG_FE_SELECT_IN_MEGA=1` the compact key array is overwritten (chosen experts logit 1.0, all others 0.0, so the Mega
  prologue selects them in slot order and the softmax of eight equal logits is exactly 0.125; all-zero keys = unrouted row);
  otherwise `topk_idx` / `topk_weights` are overwritten directly.

`DG_FE_FORCE_BALANCED=1` (or `--force-balanced`) in `tests/test_four_api_correctness.py --frontend fe` applies the same
override (`_balanced_routing`, same formula, weights, key encoding) to the same buffers, and the torch reference routes with
the overridden assignment (the frontend's own top-8 is still compared with the torch router in the ROUTER_REF line). The E2E
graph the profile times and the pipeline the test checks are therefore the same frontend + memcpy-override + Mega sequence.

## 4. Correctness sweep (frontend + fused Mega vs the torch-MoE reference)

Test: `tests/test_four_api_correctness.py --apis mxfp4_mega_moe_fused qoq_mega_moe_fused --frontend fe --reference torch-moe`
(driver `tests/fe5c_sweep.sh`, tables by `scripts/summarize_fe5c_sweep.py`). Every launch runs the frontend kernel, then the
fused Mega (SELECT_IN_MEGA effective for <= 2 rows per rank) and compares y with the torch-MoE reference. Every row is PASS.

Columns: top-8 agree = tokens whose frontend top-8 index set equals the torch router's; w max diff = max |frontend softmax
weight - torch softmax weight| over the agreeing tokens (`0` in the SELECT_IN_MEGA cells: there the compared weights are the
python decode of the keys, the Mega-prologue weights are checked through y); qoq x bytes diff = bytes of the frontend's int8 x
that differ from the torch `per_token_cast_to_int8` (rows affected, max difference in quantisation steps; mxfp4 x is always 0,
x_sf always 0 for both quants); cos / |dy| / norm ratio = kernel y vs reference; slot check = ranks whose per-(token, slot)
L2 partials all have cos >= 0.999 against the reference route.

### 4.1 10.6.131.8, one seed (20260903), SELECT_IN_MEGA 1 and 0

The two SELECT_IN_MEGA settings produced identical numbers in every cell, to all printed digits. `SELECT_IN_MEGA` line:
0 tokens differ in all launches where it applies. No `ROUTER_REF_DISAGREE` line in any launch.

| shape | routing | quant | top-8 agree | w max diff | qoq x bytes diff (rows, max dq) | cos_min | cos_mean | max abs dy | mean abs dy | norm ratio | slot check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M=2 (1/rank, owner layout ranks 0,4) | balanced | mxfp4 | 8/8 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99999678 | 0.99999940 | 0.0625 | 0.000719 | 0.999893 | 8/8 ranks clean |
| M=2 (1/rank, owner layout ranks 0,4) | balanced | qoq | 8/8 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99993002 | 0.99998266 | 0.1562 | 0.0075 | 0.999866 | 8/8 ranks clean |
| M=2 (1/rank, owner layout ranks 0,4) | normal | mxfp4 | 8/8 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99999690 | 0.99999952 | 0.0625 | 0.0009 | 0.999963 | 8/8 ranks clean |
| M=2 (1/rank, owner layout ranks 0,4) | normal | qoq | 8/8 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99993622 | 0.99998420 | 0.1953 | 0.01 | 0.999818 | 8/8 ranks clean |
| M=4 (1/rank, ranks 0,1,4,5) | balanced | mxfp4 | 8/8 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99998736 | 0.99999684 | 0.0625 | 0.00329 | 0.999869 | 8/8 ranks clean |
| M=4 (1/rank, ranks 0,1,4,5) | balanced | qoq | 8/8 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99993002 | 0.99996746 | 0.1562 | 0.0147 | 0.999891 | 8/8 ranks clean |
| M=4 (1/rank, ranks 0,1,4,5) | normal | mxfp4 | 8/8 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99999690 | 0.99999905 | 0.0625 | 0.00159 | 0.999967 | 8/8 ranks clean |
| M=4 (1/rank, ranks 0,1,4,5) | normal | qoq | 8/8 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99993628 | 0.99996907 | 0.1953 | 0.0179 | 0.999947 | 8/8 ranks clean |
| M=8 (1/rank) | balanced | mxfp4 | 8/8 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99998736 | 0.99999487 | 0.0625 | 0.00553 | 0.999884 | 8/8 ranks clean |
| M=8 (1/rank) | balanced | qoq | 8/8 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99993002 | 0.99993676 | 0.1562 | 0.0292 | 0.999956 | 8/8 ranks clean |
| M=8 (1/rank) | normal | mxfp4 | 8/8 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99999589 | 0.99999773 | 0.125 | 0.0038 | 0.999973 | 8/8 ranks clean |
| M=8 (1/rank) | normal | qoq | 8/8 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99993074 | 0.99994010 | 0.25 | 0.0382 | 0.999858 | 8/8 ranks clean |
| M=16 (T=2/rank) | balanced | mxfp4 | 16/16 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99998933 | 0.99999660 | 0.0625 | 0.00412 | 0.999947 | 8/8 ranks clean |
| M=16 (T=2/rank) | balanced | qoq | 16/16 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99992889 | 0.99993920 | 0.1875 | 0.0292 | 1.000016 | 8/8 ranks clean |
| M=16 (T=2/rank) | normal | mxfp4 | 16/16 | 0 / 5.96e-08 | 0 (0 rows, 0) | 0.99999005 | 0.99999744 | 0.125 | 0.00416 | 0.999953 | 8/8 ranks clean |
| M=16 (T=2/rank) | normal | qoq | 16/16 | 0 / 5.96e-08 | 4 (1 rows, 1) | 0.99993074 | 0.99993998 | 0.25 | 0.0381 | 0.999942 | 8/8 ranks clean |
| M=64 (T=8) | balanced | mxfp4 | 64/64 | 1.19e-07 | 0 (0 rows, 0) | 0.99996805 | 0.99999595 | 0.09375 | 0.00431 | 0.999957 | 8/8 ranks clean |
| M=64 (T=8) | balanced | qoq | 64/64 | 1.19e-07 | 37 (7 rows, 1) | 0.99991935 | 0.99993837 | 0.1875 | 0.0291 | 0.999977 | 8/8 ranks clean |
| M=64 (T=8) | normal | mxfp4 | 64/64 | 1.19e-07 | 0 (0 rows, 0) | 0.99991572 | 0.99999595 | 0.1875 | 0.00528 | 0.999934 | 8/8 ranks clean |
| M=64 (T=8) | normal | qoq | 64/64 | 1.19e-07 | 37 (7 rows, 1) | 0.99992508 | 0.99993956 | 0.375 | 0.0444 | 1.000037 | 8/8 ranks clean |
| M=128 (T=16) | balanced | mxfp4 | 128/128 | 1.19e-07 | 0 (0 rows, 0) | 0.99996328 | 0.99999577 | 0.1094 | 0.00465 | 0.999939 | 8/8 ranks clean |
| M=128 (T=16) | balanced | qoq | 128/128 | 1.19e-07 | 89 (20 rows, 1) | 0.99991882 | 0.99993807 | 0.1875 | 0.0291 | 1.000012 | 8/8 ranks clean |
| M=128 (T=16) | normal | mxfp4 | 128/128 | 1.19e-07 | 0 (0 rows, 0) | 0.99991572 | 0.99999654 | 0.25 | 0.00508 | 0.999945 | 8/8 ranks clean |
| M=128 (T=16) | normal | qoq | 128/128 | 1.19e-07 | 89 (20 rows, 1) | 0.99992102 | 0.99993944 | 0.375 | 0.0439 | 1.000026 | 8/8 ranks clean |
| M=256 (T=32) | balanced | mxfp4 | 256/256 | 1.19e-07 | 0 (0 rows, 0) | 0.99998450 | 0.99999630 | 0.07812 | 0.00427 | 0.999956 | 8/8 ranks clean |
| M=256 (T=32) | balanced | qoq | 256/256 | 1.19e-07 | 155 (37 rows, 1) | 0.99991882 | 0.99993843 | 0.2188 | 0.0291 | 0.999980 | 8/8 ranks clean |
| M=256 (T=32) | normal | mxfp4 | 256/256 | 1.19e-07 | 0 (0 rows, 0) | 0.99991572 | 0.99999666 | 0.25 | 0.00502 | 0.999952 | 8/8 ranks clean |
| M=256 (T=32) | normal | qoq | 256/256 | 1.19e-07 | 155 (37 rows, 1) | 0.99991786 | 0.99993902 | 0.5 | 0.0438 | 1.000001 | 8/8 ranks clean |
| M=256 (T=32) --slot-check | normal (FE top-8) | mxfp4 | 256/256 | 1.19e-07 | 0 (0 rows, 0) | 0.99991572 | 0.99999666 | 0.25 | 0.00502 | 0.999952 | 8/8 ranks clean |
| M=256 (T=32) --slot-check | normal (FE top-8) | qoq | 256/256 | 1.19e-07 | 155 (37 rows, 1) | 0.99991786 | 0.99993902 | 0.5 | 0.0438 | 1.000001 | 8/8 ranks clean |
| M=256 (T=32) --hot-rows 12 --slot-check | synthetic balanced + hot rows | mxfp4 | - | - | - | 0.99996305 | 0.99999601 | 0.25 | 0.0046 | 0.999935 | 8/8 ranks clean |
| M=256 (T=32) --hot-rows 12 --slot-check | synthetic balanced + hot rows | qoq | - | - | - | 0.99992698 | 0.99993992 | 0.5 | 0.0312 | 1.000069 | 8/8 ranks clean |

`5.96e-08 / 1.19e-07` = the frontend's `expf`-based softmax vs torch, 1 fp32 ulp of 0.125..0.25.

### 4.2 ComputeLab job 4252043 (viking-prod-583, 8 x H20-3e, no clock lock: correctness only), 50 seeds per cell

Same test, `--seeds 50` (seeds 20260903..20260952, same router / expert weights, new hidden rows per seed and rank),
14 cells x {SELECT_IN_MEGA 1, 0} x {normal, balanced}. Worst case over the 50 seeds per column; the two SELECT_IN_MEGA
settings agree to all printed digits, so one row per cell. No ROUTER_REF_DISAGREE anywhere: frontend top-8 == torch top-8 on
all 292 800 evaluated tokens; SELECT_IN_MEGA: 0 tokens differ in every applicable launch.

| shape | routing | quant | seeds | top-8 agree (tokens) | w max diff | qoq x bytes diff (rows, max dq) | cos_min | cos_mean (min) | max abs dy | mean abs dy (max) | norm ratio (worst) | slot check | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M=2 (1/rank, owner layout) | balanced | mxfp4 | 50 | 400/400 | 1.79e-07 | 0 (0 rows, 0) | 0.99998665 | 0.99999696 | 0.07031 | 0.00279 | 0.999789 | 8/8 ranks clean | PASS |
| M=2 (1/rank, owner layout) | balanced | qoq | 50 | 400/400 | 1.79e-07 | 223 (56 rows, 1) | 0.99992418 | 0.99998200 | 0.1719 | 0.00795 | 0.999510 | 8/8 ranks clean | PASS |
| M=2 (1/rank, owner layout) | normal | mxfp4 | 50 | 400/400 | 1.79e-07 | 0 (0 rows, 0) | 0.99997747 | 0.99999690 | 0.1875 | 0.00338 | 0.999702 | 8/8 ranks clean | PASS |
| M=2 (1/rank, owner layout) | normal | qoq | 50 | 400/400 | 1.79e-07 | 223 (56 rows, 1) | 0.99992061 | 0.99998164 | 0.4688 | 0.015 | 1.000674 | 8/8 ranks clean | PASS |
| M=4 (1/rank) | balanced | mxfp4 | 50 | 400/400 | 1.79e-07 | 0 (0 rows, 0) | 0.99996495 | 0.99999446 | 0.1094 | 0.00418 | 0.999792 | 8/8 ranks clean | PASS |
| M=4 (1/rank) | balanced | qoq | 50 | 400/400 | 1.79e-07 | 223 (56 rows, 1) | 0.99992418 | 0.99996573 | 0.1797 | 0.0156 | 0.999744 | 8/8 ranks clean | PASS |
| M=4 (1/rank) | normal | mxfp4 | 50 | 400/400 | 1.79e-07 | 0 (0 rows, 0) | 0.99994951 | 0.99999297 | 0.1953 | 0.00612 | 0.999628 | 8/8 ranks clean | PASS |
| M=4 (1/rank) | normal | qoq | 50 | 400/400 | 1.79e-07 | 223 (56 rows, 1) | 0.99992061 | 0.99996603 | 0.4688 | 0.0287 | 1.000470 | 8/8 ranks clean | PASS |
| M=8 (1/rank) | balanced | mxfp4 | 50 | 400/400 | 1.79e-07 | 0 (0 rows, 0) | 0.99996459 | 0.99999136 | 0.125 | 0.00736 | 0.999854 | 8/8 ranks clean | PASS |
| M=8 (1/rank) | balanced | qoq | 50 | 400/400 | 1.79e-07 | 223 (56 rows, 1) | 0.99992234 | 0.99993455 | 0.1875 | 0.0303 | 0.999798 | 8/8 ranks clean | PASS |
| M=8 (1/rank) | normal | mxfp4 | 50 | 400/400 | 1.79e-07 | 0 (0 rows, 0) | 0.99993587 | 0.99998701 | 0.1953 | 0.0111 | 0.999709 | 8/8 ranks clean | PASS |
| M=8 (1/rank) | normal | qoq | 50 | 400/400 | 1.79e-07 | 223 (56 rows, 1) | 0.99991846 | 0.99993372 | 0.4688 | 0.054 | 1.000335 | 8/8 ranks clean | PASS |
| M=16 (T=2) | balanced | mxfp4 | 50 | 800/800 | 2.38e-07 | 0 (0 rows, 0) | 0.99995852 | 0.99999356 | 0.125 | 0.00662 | 0.999868 | 8/8 ranks clean | PASS |
| M=16 (T=2) | balanced | qoq | 50 | 800/800 | 2.38e-07 | 398 (109 rows, 1) | 0.99991989 | 0.99993587 | 0.2188 | 0.03 | 1.000155 | 8/8 ranks clean | PASS |
| M=16 (T=2) | normal | mxfp4 | 50 | 800/800 | 2.38e-07 | 0 (0 rows, 0) | 0.99993587 | 0.99999207 | 0.1953 | 0.00837 | 0.999847 | 8/8 ranks clean | PASS |
| M=16 (T=2) | normal | qoq | 50 | 800/800 | 2.38e-07 | 398 (109 rows, 1) | 0.99991840 | 0.99993670 | 0.4688 | 0.0487 | 1.000216 | 8/8 ranks clean | PASS |
| M=64 (T=8) | balanced | mxfp4 | 50 | 3200/3200 | 0.000885 | 0 (0 rows, 0) | 0.99995816 | 0.99999470 | 0.1406 | 0.00529 | 0.999917 | 8/8 ranks clean | PASS |
| M=64 (T=8) | balanced | qoq | 50 | 3200/3200 | 0.000885 | 1643 (410 rows, 1) | 0.99991935 | 0.99993736 | 0.2109 | 0.0294 | 1.000055 | 8/8 ranks clean | PASS |
| M=64 (T=8) | normal | mxfp4 | 50 | 3200/3200 | 0.000885 | 0 (0 rows, 0) | 0.99990040 | 0.99999321 | 0.4062 | 0.00882 | 0.999808 | 8/8 ranks clean | PASS |
| M=64 (T=8) | normal | qoq | 50 | 3200/3200 | 0.000885 | 1643 (410 rows, 1) | 0.99990845 | 0.99993700 | 0.5312 | 0.0474 | 0.999893 | 8/8 ranks clean | PASS |
| M=128 (T=16) | balanced | mxfp4 | 50 | 6400/6400 | 0.000885 | 0 (0 rows, 0) | 0.99994034 | 0.99999487 | 0.1406 | 0.00529 | 0.999919 | 8/8 ranks clean | PASS |
| M=128 (T=16) | balanced | qoq | 50 | 6400/6400 | 0.000885 | 3531 (897 rows, 1) | 0.99991524 | 0.99993771 | 0.2188 | 0.0293 | 0.999954 | 8/8 ranks clean | PASS |
| M=128 (T=16) | normal | mxfp4 | 50 | 6400/6400 | 0.000885 | 0 (0 rows, 0) | 0.99975193 | 0.99999452 | 0.5 | 0.00639 | 0.999875 | 8/8 ranks clean | PASS |
| M=128 (T=16) | normal | qoq | 50 | 6400/6400 | 0.000885 | 3531 (897 rows, 1) | 0.99990284 | 0.99993742 | 0.5312 | 0.045 | 0.999929 | 8/8 ranks clean | PASS |
| M=256 (T=32) | balanced | mxfp4 | 50 | 12800/12800 | 0.00287 | 0 (0 rows, 0) | 0.99992871 | 0.99999547 | 0.1562 | 0.005 | 0.999934 | 8/8 ranks clean | PASS |
| M=256 (T=32) | balanced | qoq | 50 | 12800/12800 | 0.00287 | 6976 (1782 rows, 1) | 0.99991435 | 0.99993783 | 0.2188 | 0.0293 | 0.999958 | 8/8 ranks clean | PASS |
| M=256 (T=32) | normal | mxfp4 | 50 | 12800/12800 | 0.00287 | 0 (0 rows, 0) | 0.99975193 | 0.99999535 | 0.5 | 0.00614 | 0.999900 | 8/8 ranks clean | PASS |
| M=256 (T=32) | normal | qoq | 50 | 12800/12800 | 0.00287 | 6976 (1782 rows, 1) | 0.99990255 | 0.99993813 | 0.5312 | 0.045 | 0.999944 | 8/8 ranks clean | PASS |

Numbers reproduce across machines: the one-seed .8 cells and seed 20260903 on ComputeLab agree to all printed digits
(e.g. M=16 qoq normal cos_min 0.9999307394 / max|dy| 0.25 on both), so the sweep is deterministic and the H20-3e
(ComputeLab) vs H20 (.8) difference does not enter.

## 5. Caveats

* QoQ x vs the torch per-token int8 cast differs by exactly one quantisation step on ~0.5 elements per row (4 bytes per
  8 rows at M=8, 155 bytes per 256 rows at T=32), x_sf identical: a rounding-tie difference between the kernel's
  `v * (1/scale)` + round and torch's `x / scale` + round. It is inside the reference's tolerance (the torch reference
  dequantises the TORCH x, so this difference is counted against the kernel's y in the cos / |dy| columns above). Nothing to
  fix in the frontend unless bit-exactness with the torch cast is a requirement.
* The QoQ x is int8 stored in a `float8_e4m3fn` tensor; byte values 0x7F / 0xFF decode as NaN, so tensor-level compares
  of that buffer must be done on the raw bytes (`view(torch.uint8)`), as `fe_dump_compare.py` and `fe5_ident.py` do.
