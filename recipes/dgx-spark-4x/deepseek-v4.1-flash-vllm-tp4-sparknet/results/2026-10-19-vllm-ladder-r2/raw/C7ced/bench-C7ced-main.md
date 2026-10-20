## C7ced-main (2026-10-20T01:36:59Z)

Lane C ladder C7ced. C1,2,3,4,5,6 (C1 run 2 of 2) + prefill 2000,8000,32000,64000 (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.94 | 104.28 | 0.163 |
| C2 | 151.9 | 86.45 | 0.193 |
| C3 | 178.84 | 70.83 | 0.215 |
| C4 | 231.21 | 67.94 | 0.244 |
| C5 | 259.46 | 60.56 | 0.256 |
| C6 | 287.3 | 56.41 | 0.295 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 133.51 | 128.65 | 95.77 | 102.85 | 83.01 | 77.75 |
| json | 104.65 | 82.64 | 70.71 | 65.23 | 62.42 | 49.64 |
| narrative | 62.42 | 56.19 | 45.92 | 40.8 | 37.46 | 34.47 |
| prose | 78.55 | 65.66 | 53.33 | 48.43 | 44.63 | 40.03 |
| math | 126.9 | 96.58 | 85.13 | 81.23 | 71.93 | 76.7 |
| reasoning | 100.5 | 93.05 | 72.87 | 61.64 | 55.57 | 52.44 |
| summary | 79.1 | 59.14 | 48.7 | 45.33 | 42.98 | 37.6 |
| format | 148.59 | 109.7 | 94.18 | 98.02 | 86.52 | 82.69 |
| ceiling_count | 161.79 | 143.46 | 126.26 | 113.56 | 113.97 | 100.67 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.643 | 4587.4 |
| 8000 | 11592 | 1.992 | 5819.2 |
| 32000 | 46810 | 7.873 | 5945.8 |
| 64000 | 93335 | 15.598 | 5983.7 |
