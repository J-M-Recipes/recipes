## FINAL-main (2026-10-19T07:43:58Z)

Lane C ladder FINAL. C1,2,3,4,5,6 (C1 run 2 of 2) + prefill 2000,8000,32000,64000 (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.4 | 103.76 | 0.163 |
| C2 | 150.6 | 85.62 | 0.193 |
| C3 | 177.16 | 70.28 | 0.218 |
| C4 | 229.54 | 67.39 | 0.247 |
| C5 | 257.61 | 60.12 | 0.258 |
| C6 | 286.61 | 56.59 | 0.299 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 132.88 | 126.65 | 94.5 | 102.3 | 82.63 | 77.68 |
| json | 103.92 | 82.47 | 70.18 | 64.85 | 61.57 | 52.57 |
| narrative | 61.47 | 56.02 | 45.81 | 40.52 | 37.16 | 34.16 |
| prose | 77.43 | 65.14 | 52.98 | 47.77 | 44.44 | 39.91 |
| math | 126.91 | 95.19 | 84.01 | 80.93 | 71.39 | 76.71 |
| reasoning | 100.49 | 92.08 | 72.41 | 61.28 | 55.22 | 51.95 |
| summary | 78.6 | 58.85 | 49.21 | 43.42 | 42.91 | 37.53 |
| format | 148.35 | 108.6 | 93.16 | 98.09 | 85.66 | 82.2 |
| ceiling_count | 160.49 | 142.18 | 125.49 | 113.42 | 112.97 | 99.92 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.742 | 3977.6 |
| 8000 | 11592 | 2.035 | 5696.8 |
| 32000 | 46810 | 7.74 | 6047.8 |
| 64000 | 93335 | 15.848 | 5889.3 |
