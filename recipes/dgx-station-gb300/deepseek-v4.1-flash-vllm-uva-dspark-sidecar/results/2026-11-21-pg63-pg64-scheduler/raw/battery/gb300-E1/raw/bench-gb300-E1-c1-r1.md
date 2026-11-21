## gb300-E1-c1-r1 (2026-11-21T12:55:37Z)

dsv41-5way gb300-E1 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 458.94 | 554.96 | 0.167 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 827.07 |
| json | 540.58 |
| narrative | 311.03 |
| prose | 334.27 |
| math | 643.51 |
| reasoning | 568.68 |
| summary | 336.5 |
| format | 878.05 |
| ceiling_count | 1016.27 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
