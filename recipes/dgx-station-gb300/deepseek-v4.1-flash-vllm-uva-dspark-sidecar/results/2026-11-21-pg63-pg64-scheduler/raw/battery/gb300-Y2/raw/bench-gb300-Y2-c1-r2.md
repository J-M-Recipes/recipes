## gb300-Y2-c1-r2 (2026-11-21T16:43:56Z)

dsv41-5way gb300-Y2 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 449.57 | 536.3 | 0.098 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.96 |
| json | 534.07 |
| narrative | 311.64 |
| prose | 334.63 |
| math | 628.51 |
| reasoning | 559.17 |
| summary | 385.41 |
| format | 743.0 |
| ceiling_count | 1039.37 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
