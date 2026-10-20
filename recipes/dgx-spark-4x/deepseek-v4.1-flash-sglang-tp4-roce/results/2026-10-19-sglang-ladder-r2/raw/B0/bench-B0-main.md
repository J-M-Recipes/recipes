## B0-main (2026-10-20T00:04:45Z)

Lane B R2 B0. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 83.49 | 96.96 | 0.227 |
| C2 | 128.71 | 79.08 | 0.372 |
| C3 | 164.53 | 66.31 | 0.335 |
| C4 | 206.01 | 61.02 | 0.287 |
| C5 | 228.69 | 55.55 | 0.365 |
| C6 | 249.91 | 50.29 | 0.34 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 133.92 | 104.67 | 84.21 | 97.66 | 78.1 | 66.58 |
| json | 105.84 | 83.61 | 70.69 | 63.26 | 54.56 | 55.67 |
| narrative | 60.81 | 42.95 | 39.35 | 36.64 | 32.8 | 28.84 |
| prose | 74.5 | 59.48 | 49.03 | 37.69 | 39.87 | 34.11 |
| math | 124.54 | 88.01 | 80.92 | 69.31 | 69.12 | 60.55 |
| reasoning | 80.73 | 74.9 | 62.86 | 64.35 | 57.44 | 49.38 |
| summary | 66.04 | 52.81 | 41.33 | 42.53 | 36.73 | 32.33 |
| format | 129.32 | 126.24 | 102.13 | 76.73 | 75.77 | 74.86 |
| ceiling_count | 151.08 | 132.97 | 120.9 | 107.71 | 101.96 | 99.28 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.856 | 3444.5 |
| 8000 | 11592 | 2.095 | 5532.9 |
| 32000 | 46810 | 9.166 | 5106.7 |
| 64000 | 93335 | 18.655 | 5003.1 |
