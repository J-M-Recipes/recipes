## gb300-pg62-A3-c1-r2 (2026-11-21T12:40:34Z)

dsv41-5way gb300-pg62-A3 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 447.53 | 545.54 | 0.101 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 791.89 |
| json | 568.46 |
| narrative | 311.92 |
| prose | 334.59 |
| math | 626.24 |
| reasoning | 559.46 |
| summary | 402.68 |
| format | 769.07 |
| ceiling_count | 1053.71 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
