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
