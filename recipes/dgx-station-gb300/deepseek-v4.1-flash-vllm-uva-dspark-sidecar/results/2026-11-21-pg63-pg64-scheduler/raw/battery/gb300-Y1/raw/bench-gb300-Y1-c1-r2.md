## gb300-Y1-c1-r2 (2026-11-21T16:22:39Z)

dsv41-5way gb300-Y1 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 452.12 | 540.99 | 0.1 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.17 |
| json | 567.32 |
| narrative | 301.86 |
| prose | 335.06 |
| math | 602.57 |
| reasoning | 559.59 |
| summary | 348.17 |
| format | 819.2 |
| ceiling_count | 1016.5 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
