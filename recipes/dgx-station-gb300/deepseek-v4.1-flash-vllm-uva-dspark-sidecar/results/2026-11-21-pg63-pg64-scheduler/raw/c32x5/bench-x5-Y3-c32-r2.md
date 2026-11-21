## x5-Y3-c32-r2 (2026-11-21T17:49:11Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3663.31 | 150.39 | 0.2 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.84 |
| json | 142.93 |
| narrative | 82.28 |
| prose | 118.5 |
| math | 196.44 |
| reasoning | 153.7 |
| summary | 104.63 |
| format | 197.78 |
| ceiling_count | 224.83 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
