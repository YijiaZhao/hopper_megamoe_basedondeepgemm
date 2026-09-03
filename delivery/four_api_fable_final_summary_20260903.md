# Final Four-API H20 Delivery — Fable Frontend

Date: 2026-09-03

- Host: `10.6.131.7` (`H20-GPU-07`)
- GPU: 8 x NVIDIA H20-3e
- SM clock: explicitly locked at `1830 MHz` (`5056/5056` monitor samples)
- Container: `four_api_build`
- GitHub branch: `delivery/four-api-fable-h20-20260903`
- Workspace: `/raid/kimi/DeepGEMM_four_api_fable_h20_git`
- Reports: `/raid/kimi/megamoe_opt2_results/four_api_fable_final24_gpu07_locked1830_v2_20260903`
- Local copy: `/Users/kimiz/Downloads/BYTEDANCE_MEGAMOE/FINAL24_LOCKED1830_20260903`

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
| MXFP4 | 2 | 13.920 | 13.728 | 86.368 | 181.311 | **100.384** | **194.911** | 67.552 | 85.312 |
| MXFP4 | 8 | 13.536 | 13.728 | 109.440 | 122.207 | **123.296** | **135.999** | 131.584 | 96.287 |
| MXFP4 | 16 | 13.440 | 13.728 | 124.704 | 142.015 | **138.432** | **155.807** | 122.144 | 122.880 |
| QOQ | 2 | 14.112 | 14.496 | 95.328 | 96.000 | **110.112** | **110.592** | 67.040 | 55.744 |
| QOQ | 8 | 14.400 | 14.336 | 89.791 | 85.952 | **104.416** | **100.384** | 92.928 | 98.208 |
| QOQ | 16 | 14.688 | 14.592 | 131.296 | 138.176 | **146.752** | **152.928** | 123.584 | 119.872 |

`Frontend Fused` and `Frontend Split` are measured from their corresponding
E2E reports. Both use the same Fable frontend implementation; the small
difference is run-to-run measurement variation.

See `four_api_fable_timeline_table_20260903.md` for all component spans and
`four_api_fable_timeline_last3_20260903.md` for the final-three samples.
