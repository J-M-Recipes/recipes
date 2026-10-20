## C8pp2-main (2026-10-20T01:47:35Z)

Lane C ladder C8pp2. C1,4,6 (C1 run 2 of 2) + prefill 8000,64000 (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 93.13 | 104.52 | 0.161 |
| C4 | 211.68 | 62.11 | 0.281 |
| C6 | 257.77 | 51.81 | 0.364 |

### Per-stream tok/s by category

| category | C1 | C4 | C6 |
|---|---|---|---|
| coding | 132.77 | 86.52 | 70.57 |
| json | 104.85 | 65.29 | 50.41 |
| narrative | 62.26 | 39.75 | 33.69 |
| prose | 78.97 | 47.27 | 38.71 |
| math | 127.62 | 74.0 | 68.22 |
| reasoning | 101.47 | 58.39 | 50.46 |
| summary | 79.27 | 42.81 | 34.57 |
| format | 148.98 | 82.82 | 67.83 |
| ceiling_count | 161.73 | 104.08 | 87.65 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 2.102 | 5514.3 |
| 64000 | 93335 | 15.827 | 5897.2 |
