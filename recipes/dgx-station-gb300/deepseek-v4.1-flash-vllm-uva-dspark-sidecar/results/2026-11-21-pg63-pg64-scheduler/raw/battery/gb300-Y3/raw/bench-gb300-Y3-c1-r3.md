## gb300-Y3-c1-r3 (2026-11-21T17:41:10Z)

dsv41-5way gb300-Y3 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 477.62 | 545.81 | 0.035 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.14 |
| json | 547.55 |
| narrative | 310.93 |
| prose | 333.96 |
| math | 651.34 |
| reasoning | 556.47 |
| summary | 379.45 |
| format | 792.67 |
| ceiling_count | 1050.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
