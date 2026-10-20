## C0ctl2-main (2026-10-20T01:21:46Z)

Lane C ladder C0ctl2. C1,2,3,4,5,6 (C1 run 2 of 2) + prefill 2000,8000,32000,64000 (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.98 | 104.3 | 0.161 |
| C2 | 151.7 | 86.3 | 0.192 |
| C3 | 178.09 | 70.61 | 0.221 |
| C4 | 230.94 | 67.72 | 0.236 |
| C5 | 259.35 | 60.54 | 0.255 |
| C6 | 288.94 | 56.8 | 0.301 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 133.16 | 128.66 | 94.97 | 102.63 | 82.8 | 77.63 |
| json | 104.46 | 82.69 | 70.86 | 64.96 | 62.54 | 51.62 |
| narrative | 62.02 | 56.05 | 45.97 | 40.69 | 37.51 | 34.45 |
| prose | 78.88 | 65.3 | 53.12 | 48.35 | 44.64 | 40.04 |
| math | 126.77 | 96.41 | 84.84 | 81.25 | 71.67 | 76.74 |
| reasoning | 101.33 | 93.15 | 72.55 | 61.47 | 55.65 | 54.06 |
| summary | 78.78 | 59.06 | 48.95 | 44.33 | 43.0 | 37.71 |
| format | 148.99 | 109.08 | 93.65 | 98.06 | 86.52 | 82.19 |
| ceiling_count | 161.05 | 143.26 | 126.39 | 113.16 | 113.89 | 100.61 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.641 | 4600.3 |
| 8000 | 11592 | 2.013 | 5759.0 |
| 32000 | 46810 | 7.727 | 6058.0 |
| 64000 | 93335 | 15.832 | 5895.4 |
