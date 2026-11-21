## gb300-pg63-K2-c1-r2 (2026-11-21T17:03:34Z)

dsv41-5way gb300-pg63-K2 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 454.28 | 544.1 | 0.097 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 792.49 |
| json | 585.84 |
| narrative | 311.18 |
| prose | 334.59 |
| math | 623.09 |
| reasoning | 558.18 |
| summary | 382.75 |
| format | 764.7 |
| ceiling_count | 1048.09 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
