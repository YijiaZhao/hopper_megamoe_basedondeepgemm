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

Performance aggregation: on each GPU take the median of the final three target
executions, then take the median across eight GPUs. Split MegaMoE is
the span from L1 start through L2 end. E2E target span is Fable frontend start
through MegaMoE end. L2 flush and TP collectives are excluded.

See `four_api_fable_timeline_table_20260903.md` for the final table.
