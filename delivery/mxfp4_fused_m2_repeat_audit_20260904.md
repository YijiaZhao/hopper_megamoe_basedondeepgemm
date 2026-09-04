# MXFP4 Mega-only Fused M2 repeat audit

The single 24-matrix report entered a slow execution mode and produced
`107.808 us` on GPU 0. Three additional independent timelines were captured
with all non-target GPU containers paused and no external GPU-process
violations. Each value below is the median of that timeline's final three
complete fused-kernel spans.

| Timeline | Median (us) |
|---|---:|
| Repeat 1 | 76.096 |
| Repeat 2 | 62.432 |
| Repeat 3 | 62.432 |
| **Median across timelines** | **62.432** |

The customer performance table therefore reports `62.432 us` for this point.
The canonical 24-report customer package replaces the slow-mode report with
Repeat 2, renamed to `mega_fused_mxfp4_M2.nsys-rep`. The other repeat files were
removed from the customer package so that it contains exactly 24 timelines.
