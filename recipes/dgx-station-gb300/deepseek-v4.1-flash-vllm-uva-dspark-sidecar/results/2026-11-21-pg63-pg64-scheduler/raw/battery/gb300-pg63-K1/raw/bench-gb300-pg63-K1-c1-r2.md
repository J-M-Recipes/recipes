## gb300-pg63-K1-c1-r2 (2026-11-21T16:01:40Z)

dsv41-5way gb300-pg63-K1 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 474.37 | 544.66 | 0.036 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.35 |
| json | 543.99 |
| narrative | 310.76 |
| prose | 334.8 |
| math | 628.46 |
| reasoning | 553.2 |
| summary | 365.19 |
| format | 826.57 |
| ceiling_count | 1016.54 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
