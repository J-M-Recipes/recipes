## gb300-pg62-A7-c1-r2 (2026-11-21T14:54:42Z)

dsv41-5way gb300-pg62-A7 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 446.28 | 546.13 | 0.103 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.53 |
| json | 479.06 |
| narrative | 311.78 |
| prose | 335.14 |
| math | 632.72 |
| reasoning | 558.15 |
| summary | 386.56 |
| format | 871.07 |
| ceiling_count | 1052.99 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
