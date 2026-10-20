## B3-main (2026-10-20T03:47:12Z)

Lane B R2 B3. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 83.67 | 96.6 | 0.221 |
| C2 | 131.87 | 81.22 | 0.323 |
| C3 | 180.05 | 71.33 | 0.272 |
| C4 | 208.83 | 63.25 | 0.326 |
| C5 | 228.3 | 55.56 | 0.346 |
| C6 | 259.41 | 53.2 | 0.355 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 129.0 | 111.55 | 100.94 | 104.37 | 87.01 | 85.71 |
| json | 105.41 | 81.85 | 72.08 | 61.82 | 57.27 | 52.49 |
| narrative | 61.52 | 43.18 | 39.2 | 36.15 | 32.07 | 29.36 |
| prose | 73.79 | 56.6 | 48.49 | 38.48 | 37.52 | 32.63 |
| math | 126.28 | 101.13 | 91.16 | 71.36 | 65.54 | 63.45 |
| reasoning | 80.61 | 74.45 | 62.73 | 62.37 | 53.82 | 51.1 |
| summary | 66.16 | 56.29 | 43.88 | 45.34 | 35.39 | 33.03 |
| format | 130.05 | 124.73 | 112.19 | 86.08 | 75.84 | 77.8 |
| ceiling_count | 151.29 | 131.35 | 117.6 | 113.66 | 106.19 | 93.18 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.748 | 3941.7 |
| 8000 | 11592 | 1.985 | 5838.8 |
| 32000 | 46810 | 13.444 | 3481.9 |
| 64000 | 93335 | 21.403 | 4360.8 |
