# H20 Four-API Delivery Validation — 2026-09-03

## Source

- Repository: `hopper_megamoe_basedondeepgemm`
- Branch: `integrate/four-api-20260903`
- Kernel-clean baseline: `8cf54725663ed140a89dcd72657e31668934acb9`
- Timeline driver commits: `e3bb233`, `196b9fb`
- Machine: `10.6.131.8` (`8 x NVIDIA H20-3e`)
- Container workspace: `/raid/kimi/DeepGEMM_four_api_h20`

## Correctness rerun

The full build, same-process four-API smoke test, and exact quantized-reference
correctness test completed with exit code 0.

| API | max abs | mean abs | min cosine | mean cosine | norm ratio |
|---|---:|---:|---:|---:|---:|
| `mxfp4_mega_moe_split` | 0.0625 | 0.0047555622 | 0.9999898076 | 0.9999959469 | 0.9999283552 |
| `qoq_mega_moe_split` | 0.15625 | 0.033552093 | 0.9999098182 | 0.9999209046 | 0.9970810413 |
| `mxfp4_mega_moe_fused` | 0.0625 | 0.0047725799 | 0.9999899864 | 0.9999959469 | 0.9999219179 |
| `qoq_mega_moe_fused` | 0.140625 | 0.028386369 | 0.9999374151 | 0.9999402761 | 0.9999618530 |

Thresholds: `cos_min >= 0.99`, norm ratio in `[0.97, 1.03]`.

## 24 timeline matrix

The delivery contains all combinations of:

- precision: MXFP4 and QoQ;
- backend: Split and Fused;
- scope: E2E and MegaMoE-only;
- global token count: M2, M8, and M16.

Total: `2 x 2 x 2 x 3 = 24` Nsight Systems reports.

Remote reports:

```text
/raid/kimi/megamoe_opt2_results/four_api_delivery_8cf5472_20260903
```

Local organized delivery:

```text
/Users/kimiz/Downloads/BYTEDANCE_MEGAMOE_FOUR_API_8cf5472
```

Automated report inspection passed all 24 reports. It verifies report size,
expected frontend and MegaMoE graph kernels, and that E2E NCCL kernels remain
outside the frontend+MegaMoE CUDA graph.

Checksums and inspection details:

```text
delivery/four_api_timeline_SHA256SUMS_20260903.txt
delivery/four_api_timeline_VERIFY_20260903.json
```

## Target kernels and timing definition

Earlier handoffs documented the overall E2E scope and that Split uses separate
L1/L2 launches, but did not contain one consolidated per-report target-symbol
table.  The delivery uses the following explicit definition:

| API/backend | Target kernel or span |
|---|---|
| MXFP4 Fused | `sm90_mxfp4_mega_moe_h20_fused_impl` duration |
| QoQ Fused | `sm90_qoq_mega_moe_h20_fused_impl` duration |
| MXFP4 Split | `sm90_mxfp4_mega_moe_l1_impl` start through `sm90_mxfp4_mega_moe_l2_impl` end |
| QoQ Split | Same Split L1/L2 symbols, compiled with explicit QoQ mode; L1 start through L2 end |
| MXFP4 frontend | `router_kernel` for M2/M8 or `router_quant_topk_kernel` for M16 |
| QoQ frontend | `sm90_bf16_gemm_impl` start through `qoq_quant_topk_kernel` end |

For every number, each GPU contributes the median of its final three target
executions, followed by an arithmetic mean across eight GPUs.  E2E target span
runs from frontend start through MegaMoE completion.  TP ReduceScatter,
TP AllGather, and the 256 MiB L2 eviction kernel remain visible in the report
but are excluded from the target span.

## Timeline table

Units are microseconds.

| Scope | Precision | M | Backend | Frontend | L1 | L2 | Mega span | Target span |
|---|---|---:|---|---:|---:|---:|---:|---:|
| E2E | MXFP4 | 2 | Fused | 2.848 | - | - | 87.324 | 90.440 |
| E2E | MXFP4 | 2 | Split | 2.992 | 43.124 | 63.820 | 107.464 | 110.528 |
| E2E | MXFP4 | 8 | Fused | 2.860 | - | - | 91.952 | 95.100 |
| E2E | MXFP4 | 8 | Split | 2.984 | 64.416 | 51.216 | 119.040 | 122.036 |
| E2E | MXFP4 | 16 | Fused | 13.584 | - | - | 161.356 | 175.104 |
| E2E | MXFP4 | 16 | Split | 13.588 | 80.332 | 66.932 | 148.740 | 162.456 |
| E2E | QoQ | 2 | Fused | 184.940 | - | - | 86.736 | 271.716 |
| E2E | QoQ | 2 | Split | 184.840 | 80.864 | 53.160 | 134.444 | 319.244 |
| E2E | QoQ | 8 | Fused | 184.436 | - | - | 98.188 | 282.832 |
| E2E | QoQ | 8 | Split | 184.216 | 51.840 | 43.168 | 96.232 | 280.836 |
| E2E | QoQ | 16 | Fused | 184.792 | - | - | 163.952 | 348.872 |
| E2E | QoQ | 16 | Split | 184.352 | 70.408 | 57.040 | 126.104 | 310.664 |
| Mega-only | MXFP4 | 2 | Fused | - | - | - | 65.888 | 65.888 |
| Mega-only | MXFP4 | 2 | Split | - | 34.776 | 19.568 | 54.968 | 54.968 |
| Mega-only | MXFP4 | 8 | Fused | - | - | - | 89.704 | 89.704 |
| Mega-only | MXFP4 | 8 | Split | - | 85.232 | 35.832 | 121.664 | 121.664 |
| Mega-only | MXFP4 | 16 | Fused | - | - | - | 118.892 | 118.892 |
| Mega-only | MXFP4 | 16 | Split | - | 77.672 | 48.240 | 126.172 | 126.172 |
| Mega-only | QoQ | 2 | Fused | - | - | - | 69.516 | 69.516 |
| Mega-only | QoQ | 2 | Split | - | 33.200 | 17.856 | 51.496 | 51.496 |
| Mega-only | QoQ | 8 | Fused | - | - | - | 95.052 | 95.052 |
| Mega-only | QoQ | 8 | Split | - | 59.548 | 30.112 | 90.464 | 90.464 |
| Mega-only | QoQ | 16 | Fused | - | - | - | 122.880 | 122.880 |
| Mega-only | QoQ | 16 | Split | - | 75.288 | 38.724 | 114.704 | 114.704 |
