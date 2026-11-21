## gb300-pg63-K1-c1-r3 (2026-11-21T16:01:44Z)

dsv41-5way gb300-pg63-K1 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 489.94 | 567.06 | 0.036 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.62 |
| json | 545.51 |
| narrative | 310.99 |
| prose | 334.55 |
| math | 690.73 |
| reasoning | 558.61 |
| summary | 393.19 |
| format | 909.3 |
| ceiling_count | 1047.06 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
