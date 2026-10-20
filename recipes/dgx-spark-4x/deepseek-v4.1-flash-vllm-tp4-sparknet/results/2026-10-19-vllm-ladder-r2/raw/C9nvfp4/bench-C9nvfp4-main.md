## C9nvfp4-main (2026-10-20T01:58:58Z)

Lane C ladder C9nvfp4. C1,4,6 (C1 run 2 of 2) + prefill 8000,64000 (64000 target = 93K prompt).

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 94.3 | 106.03 | 0.163 |
| C4 | 233.18 | 68.09 | 0.238 |
| C6 | 280.77 | 54.95 | 0.294 |

### Per-stream tok/s by category

| category | C1 | C4 | C6 |
|---|---|---|---|
| coding | 138.45 | 101.3 | 73.88 |
| json | 103.77 | 66.05 | 52.0 |
| narrative | 65.07 | 40.95 | 34.44 |
| prose | 79.53 | 47.28 | 39.83 |
| math | 129.43 | 82.12 | 70.07 |
| reasoning | 100.61 | 63.15 | 53.13 |
| summary | 79.22 | 44.53 | 34.47 |
| format | 152.14 | 99.33 | 81.77 |
| ceiling_count | 165.29 | 119.92 | 100.55 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 2.106 | 5504.1 |
| 64000 | 93335 | 15.688 | 5949.3 |
