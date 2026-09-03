# MegaMoE Four-API Integration Handoff — 2026-09-03

## Objective

Starting from the clean Split baseline, integrate Fused MXFP4 and QoQ as additive, independently named modules and expose four explicit APIs:

```text
mxfp4_mega_moe_split
qoq_mega_moe_split
mxfp4_mega_moe_fused
qoq_mega_moe_fused
```

Split and Fused must retain independent workspace layouts, schedulers, namespaces, and ABI contracts.

## Current Repository State

```text
repo: /Users/kimiz/Downloads/agent/cc/hopper_megamoe_basedondeepgemm
branch: integrate/four-api-20260903
working tree: clean
```

Commits:

```text
feb2873 feat: add independent fused MXFP4 and QoQ MegaMoE APIs
b006abe fix: complete independent fused backend integration
```

Rollback/reference points:

```text
backup/four-api-pre-validation-20260903 -> feb2873
backup/four-api-validated-20260903     -> b006abe
backup/split-before-integration-20260903 -> 01c23d6
```

## Architecture

Split remains on the existing implementation:

```text
deep_gemm::layout
deep_gemm::sched
ring-buffer ABI
```

Fused uses separate modules:

```text
deep_gemm::fused_layout
deep_gemm::fused_sched
full-pool/interleaved ABI
```

Important Fused files:

```text
csrc/apis/mega_fused.hpp
csrc/jit_kernels/heuristics/sm90_fp4_mega_moe_h20_fused.hpp
csrc/jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp
deep_gemm/include/deep_gemm/comm/barrier_fused.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused.cuh
deep_gemm/include/deep_gemm/impls/sm90_fp4_mega_moe_h20_fused_body.inl
deep_gemm/include/deep_gemm/layout/mega_moe_fused.cuh
deep_gemm/include/deep_gemm/scheduler/mega_moe_fused.cuh
deep_gemm/include/deep_gemm/quantization/fp4_fused_dequant.cuh
deep_gemm/mega/fused.py
deep_gemm/quantization_mxfp4_fused.py
deep_gemm/quantization_qoq_fused.py
deep_gemm/quantization_nvfp4.py
tests/test_four_api_smoke.py
```

## H20 Target

```text
GPU: 8 x H20
SM count: 78
hidden: 3072
intermediate: 1280
experts: 384
top-k: 8
```

Remote machine and workspace:

```text
ssh root@10.6.131.8
container: nvfp4_timeline
workspace: /raid/kimi/DeepGEMM_four_api_h20
previous independently validated Fused source: /raid/kimi/DeepGEMM_fused_all_h20
```

Do not copy Fused implementation from the local H200-oriented tree. The validated H20 source of truth was `/raid/kimi/DeepGEMM_fused_all_h20`.

## API Status

The four Python/C++ APIs are present and importable.

Fused Python wrappers:

```text
deep_gemm.mxfp4_mega_moe_fused
deep_gemm.qoq_mega_moe_fused
deep_gemm.get_fused_symm_buffer_for_mega_moe
```

The Fused path has its own `FusedSymmBuffer`, weight transforms, scheduler, layout, communication barriers, and JIT bindings.

## Verified H20 Smoke

Executed on 8 x H20:

```bash
ssh root@10.6.131.8 'docker exec nvfp4_timeline bash -lc '\''
  cd /raid/kimi/DeepGEMM_four_api_h20 &&
  export DG_CUTLASS_INCLUDE_PATH=/raid/kimi/DeepGEMM_four_api_h20/third-party/cutlass/include &&
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 &&
  export PYTHONUNBUFFERED=1 &&
  torchrun --standalone --nproc_per_node=8 tests/test_four_api_smoke.py
'\'''
```

Observed output:

```text
mxfp4 fused finite 2.703125
qoq fused finite 2.765625
RC=0
```

This proves:

- both Fused kernels compile on H20;
- 8-rank execution completes;
- output tensors contain finite values;
- no crash or distributed deadlock occurred.

The two numeric values are `y.abs().mean()`. They are not latency or correctness error metrics.

## Fixes Applied During H20 Bring-Up

The Fused JIT initially failed because QoQ uses integer swap accumulators while the current PTX helper only accepted `float&`.

Added integer overloads for warpgroup operand fencing in:

```text
deep_gemm/include/deep_gemm/ptx/wgmma.cuh
```

Also added missing Fused communication/barrier support and compiler/JIT integration in commit `b006abe`.

## Validation Still Required

The integration is not fully complete from a correctness perspective.

Required next steps:

1. Run `mxfp4_mega_moe_split` explicitly on 8 x H20.
2. Run `qoq_mega_moe_split` explicitly on 8 x H20.
3. Run exact-reference correctness for Fused MXFP4.
4. Run exact-reference correctness for Fused QoQ.
5. Compare all four paths against their proper quantized reference, not only finite-output checks.
6. Record max absolute error, cosine similarity, and pass/fail tolerances.
7. Add or restore a QoQ-specific correctness test; only the MXFP4 correctness file was found in the remote four-API workspace during the last session.

Existing remote correctness test:

```text
/raid/kimi/DeepGEMM_four_api_h20/tests/test_mxfp4_mega_moe_sm90_correctness.py
```

## Existing Customer New-Shape Reference

Read:

```text
docs/HANDOFF_SM90_CUSTOMER_NEW_SHAPE_20260827.md
```

That document contains the earlier TP4/DP2/EP8 customer shape, performance commands, and strict TP pipeline correctness results.

## New Session Instructions

Start a fresh Codex process in the repository; do not resume or fork the 920K-token session.

```bash
cd /Users/kimiz/Downloads/agent/cc/hopper_megamoe_basedondeepgemm
bash /Users/kimiz/Downloads/agent/cc/run_codex.sh
```

First prompt:

```text
先读 docs/HANDOFF_FOUR_API_20260903.md 和
docs/HANDOFF_SM90_CUSTOMER_NEW_SHAPE_20260827.md。
检查 integrate/four-api-20260903 当前 Git 状态，然后继续完成四 API 的
H20 exact-reference 正确性验证。不要从旧 session 或 H200 树复制实现。
```

For forensic lookup only, the oversized old rollout is:

```text
/Users/kimiz/.codex/sessions/2026/08/31/rollout-2026-08-31T17-00-30-01a0570c-7805-7360-9041-fb6a2359dd69.jsonl
```

Do not ask the new session to read that entire JSONL; it is large enough to recreate the same context problem. Use targeted `grep` only if the handoff and Git history do not contain a specific command or result.

## 2026-09-03 Explicit Four-Kernel Follow-Up

The initial integration exposed four Python names, but the Split names were
aliases and Split QoQ still selected code generation through the process-global
`DG_W4A8_INT` environment variable.  The follow-up change makes both Split
backends explicit at the C++/pybind boundary:

```text
_C.mxfp4_mega_moe_split  -> qoq_mode=false
_C.qoq_mega_moe_split    -> qoq_mode=true
_C.mxfp4_mega_moe_fused  -> fused mode=MXFP4
_C.qoq_mega_moe_fused    -> fused mode=QoQ
```

The Split JIT runtime now carries `qoq_mode` in its arguments and uses it in the
JIT source/cache key, so MXFP4 and QoQ can be invoked sequentially in the same
process without changing `DG_W4A8_INT`.  Legacy `mxfp4_mega_moe` and
`int4_mega_moe` behavior remains available for compatibility.

Added validation:

```text
tests/test_four_api_smoke.py       # invokes all four APIs in one torchrun
tests/test_four_api_correctness.py # exact quantized/dequantized references
scripts/run_four_api_h20_validation.sh
```

The updated files were staged into the H20 container workspace:

```text
root@10.6.131.8
container: nvfp4_timeline
workspace: /raid/kimi/DeepGEMM_four_api_h20
backup: delivery/backups/20260903-explicit-four-api
```

Per the long-task policy, the build and JIT-heavy validation were prepared but
not launched automatically.  Run inside the container:

```bash
cd /raid/kimi/DeepGEMM_four_api_h20
bash scripts/run_four_api_h20_validation.sh all
```

## Final H20 Four-API Correctness — 2026-09-03

Executed on `10.6.131.8`, container `nvfp4_timeline`, using all eight H20-3e GPUs:

```bash
cd /raid/kimi/DeepGEMM_four_api_h20
bash scripts/run_four_api_h20_validation.sh all
```

The build, same-process four-API smoke test, and exact quantized-reference test
all completed with exit code 0.

| API | max abs | mean abs | min cosine | mean cosine | norm ratio |
|---|---:|---:|---:|---:|---:|
| `mxfp4_mega_moe_split` | 0.0625 | 0.0047555622 | 0.9999898076 | 0.9999959469 | 0.9999283552 |
| `qoq_mega_moe_split` | 0.15625 | 0.033552093 | 0.9999098182 | 0.9999209046 | 0.9970810413 |
| `mxfp4_mega_moe_fused` | 0.0625 | 0.0047725799 | 0.9999899864 | 0.9999959469 | 0.9999219179 |
| `qoq_mega_moe_fused` | 0.140625 | 0.028386369 | 0.9999374151 | 0.9999402761 | 0.9999618530 |

Acceptance thresholds were `cos_min >= 0.99` and norm ratio in `[0.97, 1.03]`.
All four APIs passed and produced finite output.  Full captured results are in
`delivery/four_api_correctness_h20_20260903.txt`.
