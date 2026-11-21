## gb300-E1-c1-r2 (2026-11-21T12:55:43Z)

dsv41-5way gb300-E1 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 453.6 | 545.03 | 0.099 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.09 |
| json | 559.82 |
| narrative | 311.55 |
| prose | 335.51 |
| math | 629.07 |
| reasoning | 558.63 |
| summary | 381.33 |
| format | 791.26 |
| ceiling_count | 1013.7 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
