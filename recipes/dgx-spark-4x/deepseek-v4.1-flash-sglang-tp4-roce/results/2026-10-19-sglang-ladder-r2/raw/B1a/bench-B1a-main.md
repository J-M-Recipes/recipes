## B1a-main (2026-10-20T02:38:25Z)

Lane B R2 B1a. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 82.44 | 96.16 | 0.235 |
| C2 | 127.35 | 78.52 | 0.35 |
| C3 | 173.47 | 68.99 | 0.29 |
| C4 | 203.84 | 61.6 | 0.334 |
| C5 | 233.33 | 56.38 | 0.363 |
| C6 | 252.81 | 52.01 | 0.366 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 134.05 | 111.23 | 96.47 | 97.09 | 77.16 | 86.21 |
| json | 105.03 | 80.92 | 69.11 | 59.58 | 59.61 | 53.96 |
| narrative | 61.04 | 43.08 | 39.02 | 35.4 | 32.97 | 30.58 |
| prose | 73.98 | 56.12 | 45.99 | 39.89 | 37.99 | 33.81 |
| math | 114.05 | 88.64 | 85.26 | 69.77 | 70.18 | 62.68 |
| reasoning | 85.95 | 75.06 | 64.51 | 64.39 | 52.57 | 48.9 |
| summary | 65.78 | 48.6 | 44.67 | 42.42 | 39.05 | 32.73 |
| format | 129.36 | 124.47 | 106.87 | 84.3 | 81.49 | 67.2 |
| ceiling_count | 151.15 | 132.0 | 120.09 | 113.29 | 107.36 | 96.2 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.924 | 3192.1 |
| 8000 | 11592 | 2.307 | 5024.3 |
| 32000 | 46810 | 8.48 | 5520.0 |
| 64000 | 93335 | 15.963 | 5846.8 |
