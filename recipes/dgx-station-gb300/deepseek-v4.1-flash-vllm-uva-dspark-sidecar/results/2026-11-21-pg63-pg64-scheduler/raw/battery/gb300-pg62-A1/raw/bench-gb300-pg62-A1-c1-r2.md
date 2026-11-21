## gb300-pg62-A1-c1-r2 (2026-11-21T10:51:26Z)

dsv41-5way gb300-pg62-A1 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 482.72 | 560.76 | 0.035 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 791.42 |
| json | 561.8 |
| narrative | 310.31 |
| prose | 334.66 |
| math | 661.32 |
| reasoning | 557.54 |
| summary | 362.96 |
| format | 906.1 |
| ceiling_count | 1046.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
