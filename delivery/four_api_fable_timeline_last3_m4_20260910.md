# M=4 official capture — perf/phase-stamps-probe @ 5abf9e1 (2026-09-10 ~01:30 CST), two passes

Method as r4/r5 (GPU0 last-3 median = customer; skew-free = min over 8 devices). TOKENS_LIST=4 support added to
scripts/capture_four_api_h20_timelines.sh (default unchanged). local_tokens() fixed for M=4 (owner ranks 0,1,4,5).
Raw: /raid/kimi/results/probe_branch_official_m4{,_pass2}

| Mega-only Fused (us) | customer GPU0 p1 / p2 | skew-free p1 / p2 | start skew p1 / p2 |
|---|---|---|---|
| MXFP4 M4 | 98.4* / 49.4 | 47.7 / 46.3 | 50.4* / 7.2 |
| QOQ M4   | 45.4 / 48.9 | 42.5 / 45.7 | 6.9 / 8.9 |
For reference (r5): MXFP4 M2 39.6 / M8 54.6 skew-free; QOQ M2 38.4 / M8 53.1.
\* same ~50 us launch-skew artifact as r5 MXFP4 M2.

E2E (customer GPU0): FE Fused 13.5-13.8 (MXFP4) / 14.4-14.5 (QOQ); E2E Fused MXFP4 88.3 / 85.6; QOQ 124.6 / 125.3 (heavy skew 50-214 us in both passes;
in-graph fused kernel skew-free ~60-62 us for both precisions).

In-kernel probe M=4 (rank0, us): first math 10.6-10.7, last L1 34, last L2 40-41, kernel end 47-48; one 24-K-block L1 task per SM (~22.7 us), L2 tail ~6, combine ~7.
