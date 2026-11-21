## gb300-pg63-K2-c16-r2 (2026-11-21T17:03:55Z)

dsv41-5way gb300-pg63-K2 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2399.86 | 211.81 | 0.241 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 291.01 |
| json | 196.08 |
| narrative | 120.42 |
| prose | 160.38 |
| math | 279.67 |
| reasoning | 218.1 |
| summary | 156.96 |
| format | 271.89 |
| ceiling_count | 310.51 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
