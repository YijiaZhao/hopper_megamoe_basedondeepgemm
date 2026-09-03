# Final Four-API H20 Delivery — Fable Frontend

Date: 2026-09-03

- Host: `10.6.131.7` (`H20-GPU-07`)
- GPU: 8 x NVIDIA H20-3e
- Container: `four_api_build`
- GitHub branch: `delivery/four-api-fable-h20-20260903`
- Workspace: `/raid/kimi/DeepGEMM_four_api_fable_h20_git`
- Reports: `/raid/kimi/megamoe_opt2_results/four_api_fable_final24_gpu07_rerun_20260903`
- Local copy: `/Users/kimiz/Downloads/BYTEDANCE_MEGAMOE_FABLE_FINAL24_GPU07_20260903`

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

| Precision | M | Frontend Fused | Frontend Split | E2E Fused | E2E Split | Mega Fused | Mega Split |
|---|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 13.600 | 13.728 | 134.111 | 137.247 | 65.120 | 53.087 |
| MXFP4 | 8 | 13.664 | 13.760 | 151.903 | 121.984 | 88.191 | 96.607 |
| MXFP4 | 16 | 13.824 | 13.792 | 142.559 | 158.975 | 116.543 | 126.400 |
| QOQ | 2 | 14.432 | 14.464 | 101.536 | 109.664 | 73.664 | 107.967 |
| QOQ | 8 | 14.240 | 14.368 | 115.807 | 112.575 | 93.824 | 122.016 |
| QOQ | 16 | 14.591 | 14.336 | 158.751 | 157.855 | 140.127 | 114.335 |

`Frontend Fused` and `Frontend Split` are measured from their corresponding
E2E reports. Both use the same Fable frontend implementation; the small
difference is run-to-run measurement variation.

See `four_api_fable_timeline_table_20260903.md` for all component spans and
`four_api_fable_timeline_last3_20260903.md` for the final-three samples.
