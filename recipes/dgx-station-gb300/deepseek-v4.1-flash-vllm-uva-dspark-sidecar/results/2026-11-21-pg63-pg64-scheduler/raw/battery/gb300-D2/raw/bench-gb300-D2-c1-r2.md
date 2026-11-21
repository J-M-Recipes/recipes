## gb300-D2-c1-r2 (2026-11-21T12:25:33Z)

dsv41-5way gb300-D2 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 453.32 | 549.23 | 0.101 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.73 |
| json | 519.16 |
| narrative | 311.83 |
| prose | 334.76 |
| math | 630.63 |
| reasoning | 559.07 |
| summary | 374.92 |
| format | 868.7 |
| ceiling_count | 1020.66 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
