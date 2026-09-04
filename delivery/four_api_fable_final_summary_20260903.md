# Final Four-API H20 Delivery — Fable Frontend

Date: 2026-09-04

- Host: `10.6.131.7` (`H20-GPU-07`)
- GPU: 8 x NVIDIA H20-3e
- SM clock: explicitly locked at `1830 MHz` (`4968/4968` monitor samples; 250 ms process audit: 0 violations)
- Container: `four_api_build`
- GitHub branch: `delivery/four-api-fable-h20-20260903`
- Workspace: `/raid/kimi/DeepGEMM_four_api_fable_h20_git`
- Reports: `/raid/kimi/megamoe_opt2_results/four_api_fable_final24_gpu07_exclusive_v2_locked1830_20260904`
- Local copy: `/Users/kimiz/Downloads/BYTEDANCE_MEGAMOE/FINAL24_EXCLUSIVE_LOCKED1830_20260904`

All 24 reports were captured from an empty output directory on an otherwise
idle 8-GPU host. Every E2E report uses the same Fable dynamic-M frontend for
both Split and Fused backends. The frontend performs Router WMMA, activation
quantization, TopK8, and softmax in `router_quant_topk_kernel`.

Correctness gates passed for Fable frontend local M=1,2,4,8,16,32,64 and for
all four explicit MegaMoE APIs against their quantized references.

Performance aggregation: use GPU 0 / rank 0 in each report, take the final
three complete MegaMoE executions, and report their median. For Split, each
complete MegaMoE span runs from the corresponding L1 start through L2 end,
including the inter-kernel gap. For Fused, it is the complete fused-kernel
start-to-end duration. E2E target span runs from Fable frontend start through
MegaMoE completion. L2 flush and TP collectives are excluded.

## Final median latency (us)

| Precision | M | FE Fused | FE Split | E2E Mega Fused | E2E Mega Split | E2E Fused | E2E Split | Mega-only Fused | Mega-only Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 13.760 | 13.856 | 86.624 | 107.520 | **100.640** | **121.440** | 107.808 | 57.152 |
| MXFP4 | 8 | 13.632 | 13.792 | 91.520 | 125.568 | **105.440** | **139.424** | 87.584 | 105.088 |
| MXFP4 | 16 | 13.696 | 13.856 | 133.024 | 147.424 | **147.008** | **161.088** | 115.232 | 128.416 |
| QOQ | 2 | 14.272 | 14.400 | 92.896 | 87.040 | **107.584** | **101.632** | 64.192 | 50.400 |
| QOQ | 8 | 14.432 | 14.432 | 95.456 | 84.352 | **110.144** | **98.880** | 92.608 | 81.760 |
| QOQ | 16 | 14.720 | 14.400 | 138.784 | 125.120 | **153.504** | **139.968** | 123.712 | 114.112 |

`Frontend Fused` and `Frontend Split` are measured from their corresponding
E2E reports. Both use the same Fable frontend implementation; the small
difference is run-to-run measurement variation.

See `four_api_fable_timeline_table_20260903.md` for all component spans and
`four_api_fable_timeline_last3_20260903.md` for the final-three samples.
