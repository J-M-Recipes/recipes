## B0ctl-main (2026-10-20T01:13:28Z)

Lane B R2 B0ctl. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 82.91 | 96.43 | 0.23 |
| C2 | 127.1 | 76.39 | 0.271 |
| C3 | 177.95 | 69.91 | 0.271 |
| C4 | 208.16 | 60.68 | 0.287 |
| C5 | 227.13 | 54.57 | 0.315 |
| C6 | 261.6 | 52.2 | 0.347 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 134.52 | 109.61 | 97.97 | 94.11 | 75.51 | 82.3 |
| json | 105.45 | 81.21 | 66.14 | 59.61 | 55.98 | 49.65 |
| narrative | 61.36 | 43.15 | 38.5 | 33.46 | 31.0 | 27.86 |
| prose | 74.4 | 59.71 | 47.96 | 37.93 | 38.71 | 33.81 |
| math | 119.93 | 87.68 | 92.39 | 69.04 | 63.85 | 64.25 |
| reasoning | 80.38 | 73.0 | 67.6 | 63.7 | 53.82 | 51.19 |
| summary | 66.18 | 56.23 | 41.61 | 41.12 | 35.57 | 32.11 |
| format | 129.25 | 100.57 | 107.07 | 86.46 | 82.15 | 76.46 |
| ceiling_count | 151.32 | 132.42 | 120.06 | 112.08 | 107.21 | 98.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.756 | 3904.7 |
| 8000 | 11592 | 2.016 | 5750.9 |
| 32000 | 46810 | 14.904 | 3140.8 |
| 64000 | 93335 | 29.673 | 3145.4 |
