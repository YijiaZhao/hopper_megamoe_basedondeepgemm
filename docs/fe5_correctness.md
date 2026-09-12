# FE round 5 correctness: torch-MoE reference, FE layout gate, forced-balanced routing, E2E sweep

Branch `perf/fe5-correctness` (python / tests / docs only, branched from `perf/phase-stamps-probe` 92fb07b).
Machines: 10.6.131.8 (8 x H20, container `fe5c_build` = the validated `lmsysorg/sglang@sha256:687efca0...` image,
worktree `/raid/kimi/dg_fe5c`, results `/raid/kimi/results/fe5c/`), ComputeLab job 4252043 (viking-prod-583, 8 x H20-3e,
`lmsysorg/sglang:dev`, container `fe5c_cl`, repo `/work/repo`, results node-local `/tmp/kimiz_fe5c_out/`).
Kernel build under test: probe tip 92fb07b (`csrc/fable_frontend.cu` and the fused Mega unchanged by this branch).

## 1. The reference is a real-routing pure-torch MoE (confirmed; `--reference torch-moe` added as the front door)

`tests/test_four_api_correctness.py` had three reference flavours before this branch:

| flag | routing the reference uses | activations | contains our kernels? |
|---|---|---|---|
| default (`--frontend none`) | synthetic balanced (`slot * 48 + (8 g + 7 slot) % 48`, weight 1/8) written into the buffer | torch per-token cast | no, but routing is not real |
| `--frontend fe` | the FE kernel's own topk_idx / topk_weights read back from the buffer | the FE kernel's x / x_sf | the reference depends on the FE outputs |
| `--frontend fe --router-ref torch` (ae646c8) | pure torch: bf16 router GEMM (fp32 accumulate) -> bf16-rounded logits -> top-8 (value desc, index asc on ties) -> fp32 softmax over the 8 selected bf16 logits (the FE's normalisation, `topk_softmax_token`: `expf(v - max) / sum` over the selected) | torch per-token cast (`per_token_cast_to_fp8` / `per_token_cast_to_int8`) | no |

The MoE part is the same in every flavour: for every (global token, slot) owned by this rank, the reference computes
`L1 = dequant(W1[e]) @ x_ref`, clamps gate / up (`activation_clamp`), `mid = silu(gate) * up * w`, quantises `mid` to the
kernel's intermediate format (fp8 per-64 group amax/448 with power-of-two scale for MXFP4, int8 per-64 amax/127 for QoQ),
`dequant(W2[e]) @ mid` rounded to bf16, then sums the 8 routes across the 8 EP ranks (all-reduce) and rounds to bf16.
Weight dequantisation is the kernel-exact one: MXFP4 `dequantize_mxfp4_kernel_exact_fp32(e2m1, e8m0, row reference exponent)`,
QoQ `dequantize_qoq_to_fp32(int4, s1, s2)` (deep_gemm/quantization_*_fused.py). So `--frontend fe --router-ref torch` IS the
requested reference: same weights, real router routing, none of our kernels.

Added here:
* `--reference torch-moe` = `--frontend fe --router-ref torch` (implies `--frontend fe` when none is given).
* The `ROUTER_REF` line now reports, per launch: top-8 index-set agreement (tokens), weight max |diff| on agreeing tokens AND
  over all tokens (union of both expert sets, a missing expert counts as weight 0), x byte diffs (bytes, rows, max |dq| in
  quantisation steps) and x_sf word diffs; up to 4 disagreeing tokens are printed with the experts that differ and their
  torch logits (`ROUTER_REF_DISAGREE`).
* `DG_FE_SELECT_IN_MEGA=1` (the pipeline configuration, <= 2 rows per rank): the FE leaves only the 384 compact keys per
  token; the test decodes the top-8 / softmax from the key array in python (`_topk_from_keys`, inverse of the orderable-bf16
  key encoding) for the ROUTER_REF report, reads the routing the Mega prologue actually wrote into the buffer after the
  kernel, and reports `SELECT_IN_MEGA ... N tokens differ` between the two (0 in every run below).
* `--global-tokens 2 | 4` (owner layout, M < 8) now works with `--frontend fe`: every rank runs the FE, the inactive ranks'
  rows are then unrouted (topk_idx -1 / all-zero keys, exactly what `DG_PROFILE_FORCE_BALANCED` does for inactive rows).
* `--seeds N`: N seeds in one process (same expert / router weights, new hidden rows), one `RESULT api= seed=` line each and a
  `SUMMARY` line (min cos, max |dy|, norm-ratio range, top-8 agreement total, failing seeds).

## 2. FE output layout gate: `tests/fe_dump_compare.py`

Runs the FE alone (single GPU, `types.SimpleNamespace` buffer, 64 rows) for `--seeds` x rows {1, 2} x {mxfp4, qoq} under two
env configurations (one child process each, because the knobs are process-static) and byte-compares every buffer the fused
Mega consumes, after pre-filling the buffers with a sentinel so unwritten slots are visible:

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
fp32 / key -> expert) and a class: `identical`, `zeros/uninitialised` (one side holds the sentinel at every differing
element), `whole-row missing`, `rounding` (<= 1 quantisation step / 2^-20 relative), `layout shift` (row permutation or a
K128-group / scale-slot shift), `other`; then per-buffer totals with the axis hit (how many seeds, which rows, which quants).
`--ref-torch` additionally compares each dump's x / x_sf with the torch per-token casts. `DG_FE_SELECT_IN_MEGA` 0 vs 1 is
always covered (x / x_sf of the two launches are compared inside each dump).

### Result on .8, probe tip 92fb07b, `--env-a DG_FE_CC_LEAN=0 --env-b DG_FE_CC_LEAN=1 --seeds 64 --ref-torch` (log `/raid/kimi/results/fe5c_dump_lean.log`)

```
FE_DUMP_COMPARE cells=256 failing_cells=0
  x, x_sf, topk_idx, topk_weights, ticket, x_sel1, x_sf_sel1, keys, ticket_sel1: identical in all 256 cells
  A:x / B:x / A:x_sf / B:x_sf select_in_mega=0 vs 1: identical in all 256 cells
  REF_TORCH A and B: mxfp4 x / x_sf byte-identical to the torch per-token casts in every cell;
                     qoq rows=1: 12/64 cells differ (38 bytes, max |dq| 1 ulp, x_sf 0 words); rows=2: 12/64 cells (56 bytes, 1 ulp, x_sf 0)
FE_DUMP_COMPARE PASS
```

Classification of the "128 of 256 cells x/x_sf mismatch": at this tip there is NO byte difference between LEAN=0 and LEAN=1
(nor between select_in_mega 0 / 1) in any of the 9 buffers, padding rows included. The 128 / 256 pattern is one axis = the
**quant axis (all 128 qoq cells)**, and it is the harness artefact already described in docs/fe_cc_round5.md section 4: the QoQ
x is int8 stored in a `float8_e4m3fn` tensor, whose byte values 0x7F / 0xFF decode as NaN, and NaN != NaN flagged every qoq
row containing a +-127 (a near-certainty at 3072 elements) as "mismatching" -- baseline vs baseline shows the same 128. The
byte compare here (and the fixed `fe5_ident.py`, 657fa21) shows the buffers identical. What does differ, identically for both
LEAN settings, is the QoQ x against the torch `per_token_cast_to_int8`: <= 1 quantisation step on ~0.5 elements per 1000
rows-equivalent (12 of 64 cells, a handful of bytes each), x_sf identical -- an int8 rounding-tie difference (kernel
`v * (1/scale)` + round vs torch `x / scale` + round), not a layout problem; it is what the `x_bytes_diff` column of the sweep
below shows for every qoq row and it is within the reference's tolerance (cos_min 0.99992+).

## 3. Forced-balanced routing knob

The 2026-09-04 delivery table's "balanced" E2E numbers come from `DG_PROFILE_FORCE_BALANCED=1` (tests/profile_four_api_h20.py,
26b932b; used by `tests/fe5_campaign.sh`, columns `*_balanced_p*`). Exact usage:

```
DG_PROFILE_FORCE_BALANCED=1 DG_FE_CC_LEAN=1 DG_FE_SELECT_IN_MEGA=1 \
  SCOPES=e2e BACKENDS=fused QUANTS="mxfp4 qoq" TOKENS_LIST="2 4 8 16" OUT=<dir> FORCE=1 \
  bash scripts/capture_four_api_h20_timelines.sh
```

What it does (E2E scope only): the FE kernel runs exactly as in the normal mode; one graph memcpy node (`cudaMemcpyAsync`,
no kernel; inside the E2E span, outside both kernel spans) then overwrites the FE's routing output before the Mega:
* assignment (`forced_balanced_routing`): for active row t of rank r (global token g = r * local_rows + t), slot s ->
  expert `s * 48 + (g + 7 s) % 48`, i.e. one route to every EP rank, **weight 1/8 each (uniform)**; inactive owner-layout rows
  (M < 8) are unrouted;
* with `DG_FE_SELECT_IN_MEGA=1` the compact key array is overwritten (chosen experts logit 1.0, all others 0.0, so the Mega
  prologue selects them in slot order and the softmax of eight equal logits is exactly 0.125; all-zero keys = unrouted row);
  otherwise `topk_idx` / `topk_weights` are overwritten directly.

Added here for correctness runs: `DG_FE_FORCE_BALANCED=1` (or `--force-balanced`) in `tests/test_four_api_correctness.py`
with `--frontend fe`: the same override (`_balanced_routing`, same formula, same uniform 1/8 weights / key encoding), applied
to the same buffers, and the torch reference routes with the overridden assignment (the FE's own top-8 is still compared with
the torch router in the ROUTER_REF line). The E2E graph the profile times and the pipeline the test checks are therefore the
same FE + memcpy-override + Mega sequence.

## 4. Correctness sweep (FE + fused Mega vs the torch-MoE reference)

RESULTS_PLACEHOLDER

## 5. What failed / caveats

CAVEATS_PLACEHOLDER
