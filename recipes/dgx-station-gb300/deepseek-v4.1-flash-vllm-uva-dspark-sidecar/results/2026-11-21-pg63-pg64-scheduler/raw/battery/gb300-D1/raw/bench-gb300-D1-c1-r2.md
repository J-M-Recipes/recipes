## gb300-D1-c1-r2 (2026-11-21T12:10:29Z)

dsv41-5way gb300-D1 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 464.17 | 557.51 | 0.101 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 806.03 |
| json | 620.55 |
| narrative | 311.73 |
| prose | 335.11 |
| math | 647.63 |
| reasoning | 558.96 |
| summary | 307.56 |
| format | 872.54 |
| ceiling_count | 1047.64 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
