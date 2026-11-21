## x5-Y2-c32-r2 (2026-11-21T16:52:00Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3710.96 | 151.89 | 0.187 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 198.71 |
| json | 146.38 |
| narrative | 82.13 |
| prose | 114.98 |
| math | 207.16 |
| reasoning | 155.27 |
| summary | 108.38 |
| format | 202.12 |
| ceiling_count | 217.07 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
