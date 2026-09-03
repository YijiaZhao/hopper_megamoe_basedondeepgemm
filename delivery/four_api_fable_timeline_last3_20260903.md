Device: GPU 0; values are medians of the final three complete spans.

## E2E timelines

| Precision | M | Backend | Frontend median | Mega median | E2E median |
|---|---:|---|---:|---:|---:|
| MXFP4 | 2 | FUSED | 13.600 | 119.743 | **134.111** |
| MXFP4 | 2 | SPLIT | 13.728 | 123.231 | **137.247** |
| MXFP4 | 8 | FUSED | 13.664 | 137.951 | **151.903** |
| MXFP4 | 8 | SPLIT | 13.760 | 108.096 | **121.984** |
| MXFP4 | 16 | FUSED | 13.824 | 128.511 | **142.559** |
| MXFP4 | 16 | SPLIT | 13.792 | 145.375 | **158.975** |
| QOQ | 2 | FUSED | 14.432 | 87.072 | **101.536** |
| QOQ | 2 | SPLIT | 14.464 | 95.104 | **109.664** |
| QOQ | 8 | FUSED | 14.240 | 101.599 | **115.807** |
| QOQ | 8 | SPLIT | 14.368 | 98.111 | **112.575** |
| QOQ | 16 | FUSED | 14.591 | 144.191 | **158.751** |
| QOQ | 16 | SPLIT | 14.336 | 143.391 | **157.855** |

## Mega-only timelines

| Precision | M | Backend | Mega median |
|---|---:|---|---:|
| MXFP4 | 2 | FUSED | **65.120** |
| MXFP4 | 2 | SPLIT | **53.087** |
| MXFP4 | 8 | FUSED | **88.191** |
| MXFP4 | 8 | SPLIT | **96.607** |
| MXFP4 | 16 | FUSED | **116.543** |
| MXFP4 | 16 | SPLIT | **126.400** |
| QOQ | 2 | FUSED | **73.664** |
| QOQ | 2 | SPLIT | **107.967** |
| QOQ | 8 | FUSED | **93.824** |
| QOQ | 8 | SPLIT | **122.016** |
| QOQ | 16 | FUSED | **140.127** |
| QOQ | 16 | SPLIT | **114.335** |
