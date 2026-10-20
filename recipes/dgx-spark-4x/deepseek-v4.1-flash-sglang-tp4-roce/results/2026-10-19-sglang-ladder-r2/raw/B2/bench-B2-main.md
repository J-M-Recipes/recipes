## B2-main (2026-10-20T03:27:59Z)

Lane B R2 B2. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 81.16 | 94.64 | 0.242 |
| C2 | 123.02 | 77.72 | 0.393 |
| C3 | 165.97 | 67.68 | 0.339 |
| C4 | 209.68 | 62.4 | 0.31 |
| C5 | 232.83 | 56.58 | 0.324 |
| C6 | 258.59 | 53.3 | 0.392 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 131.84 | 112.6 | 90.11 | 99.93 | 87.73 | 82.45 |
| json | 93.86 | 82.19 | 69.0 | 60.05 | 53.26 | 51.76 |
| narrative | 56.24 | 43.19 | 38.21 | 35.43 | 32.82 | 29.16 |
| prose | 67.1 | 59.98 | 50.3 | 38.3 | 38.47 | 33.48 |
| math | 123.96 | 100.07 | 82.45 | 77.17 | 68.56 | 64.18 |
| reasoning | 87.95 | 75.43 | 61.49 | 64.89 | 49.26 | 50.46 |
| summary | 66.0 | 47.35 | 41.29 | 38.7 | 36.58 | 35.82 |
| format | 130.16 | 100.91 | 108.58 | 84.76 | 85.92 | 79.12 |
| ceiling_count | 150.15 | 131.98 | 121.59 | 113.18 | 107.07 | 98.39 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.016 | 2903.4 |
| 8000 | 11592 | 2.369 | 4892.9 |
| 32000 | 46810 | 8.495 | 5510.2 |
| 64000 | 93335 | 16.288 | 5730.4 |
