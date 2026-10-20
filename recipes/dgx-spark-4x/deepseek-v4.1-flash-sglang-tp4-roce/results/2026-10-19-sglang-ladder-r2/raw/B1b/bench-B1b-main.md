## B1b-main (2026-10-20T03:02:36Z)

Lane B R2 B1b. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 81.19 | 93.97 | 0.226 |
| C2 | 135.16 | 81.04 | 0.279 |
| C3 | 170.96 | 69.56 | 0.316 |
| C4 | 205.01 | 61.2 | 0.32 |
| C5 | 241.03 | 57.29 | 0.314 |
| C6 | 255.15 | 52.72 | 0.372 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 131.78 | 107.03 | 103.07 | 94.88 | 88.36 | 80.94 |
| json | 104.38 | 82.5 | 70.09 | 58.96 | 58.07 | 53.08 |
| narrative | 61.12 | 47.2 | 38.75 | 34.04 | 32.26 | 29.62 |
| prose | 70.17 | 56.59 | 47.41 | 38.38 | 39.61 | 35.04 |
| math | 112.02 | 100.52 | 91.63 | 73.9 | 66.41 | 63.45 |
| reasoning | 77.92 | 74.95 | 62.44 | 60.96 | 62.81 | 51.22 |
| summary | 64.31 | 53.31 | 46.05 | 41.5 | 35.42 | 37.42 |
| format | 130.03 | 126.22 | 97.08 | 86.97 | 75.39 | 70.95 |
| ceiling_count | 149.69 | 130.15 | 120.71 | 112.67 | 97.76 | 91.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.804 | 3667.1 |
| 8000 | 11592 | 2.001 | 5793.0 |
| 32000 | 46810 | 7.997 | 5853.4 |
| 64000 | 93335 | 15.803 | 5906.1 |
