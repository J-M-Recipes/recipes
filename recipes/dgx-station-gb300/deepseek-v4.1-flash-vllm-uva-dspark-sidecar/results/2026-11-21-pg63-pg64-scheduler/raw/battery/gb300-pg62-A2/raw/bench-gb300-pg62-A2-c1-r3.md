## gb300-pg62-A2-c1-r3 (2026-11-21T11:53:02Z)

dsv41-5way gb300-pg62-A2 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 476.39 | 540.48 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.05 |
| json | 530.89 |
| narrative | 311.63 |
| prose | 334.7 |
| math | 663.23 |
| reasoning | 557.64 |
| summary | 332.3 |
| format | 799.4 |
| ceiling_count | 1015.87 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
