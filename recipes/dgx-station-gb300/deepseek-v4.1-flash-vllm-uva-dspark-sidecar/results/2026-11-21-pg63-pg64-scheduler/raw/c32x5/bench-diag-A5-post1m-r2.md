## diag-A5-post1m-r2 (2026-11-21T14:04:17Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3732.86 | 149.68 | 0.171 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 207.71 |
| json | 141.96 |
| narrative | 79.53 |
| prose | 118.7 |
| math | 196.38 |
| reasoning | 153.31 |
| summary | 104.2 |
| format | 195.65 |
| ceiling_count | 220.78 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
