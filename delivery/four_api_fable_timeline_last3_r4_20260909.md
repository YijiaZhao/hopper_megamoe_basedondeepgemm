# Official capture r4 — branch perf/phase-stamps-probe @ 132c822 (2026-09-09 ~19:00 CST)

Method: scripts/capture_four_api_h20_timelines.sh, .7 locked 1830 MHz, GPUs exclusive, 24 nsys timelines,
scripts/summarize_four_api_h20_last3.py (GPU 0, medians of final three complete spans) = customer method.
Extra column: skew-free kernel time = min-over-8-devices duration (scripts/reconcile_nsys_devices.py), last-3 window.
Build defaults: push dispatch ON (d35f27f), lean routing ON, fine combine ON, QoQ inline s2 ON (LiquidGEMM dequant),
wgmma pipe un-serialised; stream-K / tiny-M GEMV OFF. Raw: /raid/kimi/results/probe_branch_official_r4

## Customer method (Mega-only Fused, us)

| | 09-04 delivery | r2 (09-08) | r3 (b304589) | **r4 (132c822)** | skew-free r4 | start skew r4 | target |
|---|---|---|---|---|---|---|---|
| MXFP4 M2  | 62.4  | 53.3  | 47.4  | **46.9** | 38.5 | 9.4 | 53 PASS |
| MXFP4 M8  | 87.6  | 69.3  | 68.9  | **59.1** | 54.0 | 6.2 | 61 PASS |
| MXFP4 M16 | 115.2 | 91.0  | 91.0  | **74.0** | 74.0 | 7.3 | 85 PASS |
| QOQ M2    | 64.2  | 69.2  | 84.6* | **43.8** | 39.6 | 5.6 | — |
| QOQ M8    | 92.6  | 80.0  | 66.4  | **54.9** | 54.1 | 4.9 | — |
| QOQ M16   | 123.7 | 110.8 | 86.9  | 125.2* | 73.4 | 51.7* | — |

\* launch-skew artifact in the last-3 window (QoQ M16 r4: two ranks launched ~50 us late; last-10 GPU0 median 76.3,
skew-free 73.4). Customer number = skew-free kernel time + GPU0 launch lead (host jitter, typically 5–11 us).

## Full r4 table

| Precision | M | FE Fused | FE Split | E2E Mega Fused | E2E Mega Split | E2E Fused | E2E Split | Mega-only Fused | Mega-only Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 13.760 | 13.600 | 82.528 | 118.272 | 96.832 | 131.968 | 46.944 | 53.888 |
| MXFP4 | 8 | 13.728 | 13.568 | 61.248 | 131.455 | 75.520 | 145.439 | 59.072 | 97.344 |
| MXFP4 | 16 | 13.664 | 13.600 | 94.560 | 241.440 | 108.512 | 255.424 | 73.952 | 147.776 |
| QOQ | 2 | 14.368 | 14.336 | 59.359 | 140.768 | 74.015 | 155.456 | 43.775 | 49.536 |
| QOQ | 8 | 14.432 | 14.624 | 75.872 | 110.784 | 90.336 | 125.216 | 54.912 | 88.032 |
| QOQ | 16 | 14.656 | 14.400 | 94.400 | 145.280 | 109.344 | 159.968 | 125.216* | 113.440 |

Split columns are the untouched split kernel and fluctuate with host skew (e.g. MXFP4 M16 E2E Split 255 vs 160 in r3).
