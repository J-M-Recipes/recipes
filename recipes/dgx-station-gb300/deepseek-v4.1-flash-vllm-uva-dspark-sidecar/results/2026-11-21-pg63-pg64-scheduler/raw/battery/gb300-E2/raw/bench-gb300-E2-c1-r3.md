## gb300-E2-c1-r3 (2026-11-21T13:10:45Z)

dsv41-5way gb300-E2 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 485.1 | 558.17 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 795.31 |
| json | 549.52 |
| narrative | 311.65 |
| prose | 334.69 |
| math | 649.17 |
| reasoning | 557.7 |
| summary | 370.95 |
| format | 896.4 |
| ceiling_count | 1025.64 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
