## gb300-GR-c1-r3 (2026-11-21T11:37:26Z)

dsv41-5way gb300-GR c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 482.73 | 547.51 | 0.033 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.27 |
| json | 573.05 |
| narrative | 312.16 |
| prose | 333.93 |
| math | 630.14 |
| reasoning | 557.44 |
| summary | 389.4 |
| format | 789.73 |
| ceiling_count | 1044.33 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
