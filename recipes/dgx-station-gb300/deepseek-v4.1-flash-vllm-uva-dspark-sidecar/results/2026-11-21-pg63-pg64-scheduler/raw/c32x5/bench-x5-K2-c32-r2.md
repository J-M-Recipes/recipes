## x5-K2-c32-r2 (2026-11-21T17:11:40Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3700.34 | 148.73 | 0.17 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.71 |
| json | 141.66 |
| narrative | 79.39 |
| prose | 115.5 |
| math | 187.11 |
| reasoning | 151.51 |
| summary | 106.96 |
| format | 203.01 |
| ceiling_count | 225.84 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
