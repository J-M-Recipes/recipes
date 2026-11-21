## gb300-pg63-K2-c1-r3 (2026-11-21T17:03:38Z)

dsv41-5way gb300-pg63-K2 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 482.82 | 552.16 | 0.035 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.68 |
| json | 576.36 |
| narrative | 311.74 |
| prose | 334.19 |
| math | 661.49 |
| reasoning | 555.76 |
| summary | 382.64 |
| format | 801.45 |
| ceiling_count | 1046.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
