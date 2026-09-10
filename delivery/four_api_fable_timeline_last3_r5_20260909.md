# Official capture r5 — branch perf/phase-stamps-probe @ 17db9b9 (2026-09-09 ~19:50 CST)

Same method as r4 (customer: GPU 0, median of final three spans; plus skew-free = min over 8 devices, last-3 window).
Build = r4 (132c822) + QoQ packed prefetch default (DG_FP4_QIS2_PREFETCH_PACKED=1); MXFP4 path unchanged vs r4.
Raw: /raid/kimi/results/probe_branch_official_r5

| Mega-only Fused (us) | r4 customer | **r5 customer** | r4 skew-free | **r5 skew-free** | r5 start skew | target |
|---|---|---|---|---|---|---|
| MXFP4 M2  | 46.9 | 88.3* | 38.5 | 39.6 | 47.7* | 53 |
| MXFP4 M8  | 59.1 | **56.4** | 54.0 | 54.6 | 6.3 | 61 PASS |
| MXFP4 M16 | 74.0 | **83.4** | 74.0 | 75.7 | 8.6 | 85 PASS (thin) |
| QOQ M2    | 43.8 | **45.5** | 39.6 | 38.4 | 7.4 | — |
| QOQ M8    | 54.9 | **59.3** | 54.1 | 53.1 | 7.1 | — |
| QOQ M16   | 125.2* | **74.0** | 73.4 | 74.0 | 4.6 | — |

\* launch-skew artifact inside the last-3 window (r5 MXFP4 M2: rank 0 launched ~48 us before the slowest rank in all
three rounds; r4 QoQ M16 likewise). Skew-free kernel time is stable across r4/r5 within ~1.5 us at every point:
MXFP4 39.6 / 54.6 / 75.7, QoQ 38.4 / 53.1 / 74.0. Customer number = skew-free + rank-0 launch lead (5–11 us typical).

Full r5 table

| Precision | M | FE Fused | FE Split | E2E Mega Fused | E2E Mega Split | E2E Fused | E2E Split | Mega-only Fused | Mega-only Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP4 | 2 | 13.696 | 13.472 | 60.672 | 116.863 | 74.496 | 130.367 | 88.320* | 60.160 |
| MXFP4 | 8 | 13.600 | 13.664 | 75.264 | 101.888 | 89.216 | 115.520 | 56.448 | 101.504 |
| MXFP4 | 16 | 13.696 | 13.728 | 188.895* | 187.104 | 202.655* | 200.832 | 83.424 | 125.823 |
| QOQ | 2 | 14.368 | 14.592 | 79.488 | 105.280 | 94.336 | 119.840 | 45.536 | 52.608 |
| QOQ | 8 | 14.240 | 14.400 | 116.288 | 87.136 | 130.944 | 101.568 | 59.328 | 81.536 |
| QOQ | 16 | 14.528 | 14.656 | 91.072 | 135.199 | 105.568 | 149.919 | 74.048 | 115.008 |

E2E columns are span-based and inherit host skew (MXFP4 M16 E2E 202 here vs 108 in r4 — same kernel).
