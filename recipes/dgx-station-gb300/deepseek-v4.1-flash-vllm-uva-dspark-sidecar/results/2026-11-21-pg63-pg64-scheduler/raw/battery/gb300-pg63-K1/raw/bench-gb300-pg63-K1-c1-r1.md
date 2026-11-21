## gb300-pg63-K1-c1-r1 (2026-11-21T16:01:36Z)

dsv41-5way gb300-pg63-K1 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 469.25 | 547.63 | 0.049 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 827.31 |
| json | 600.57 |
| narrative | 311.45 |
| prose | 336.07 |
| math | 614.15 |
| reasoning | 571.55 |
| summary | 380.13 |
| format | 739.83 |
| ceiling_count | 1016.88 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
