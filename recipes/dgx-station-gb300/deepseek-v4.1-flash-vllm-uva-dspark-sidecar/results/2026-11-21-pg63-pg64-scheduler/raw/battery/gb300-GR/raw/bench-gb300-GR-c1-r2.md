## gb300-GR-c1-r2 (2026-11-21T11:37:22Z)

dsv41-5way gb300-GR c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 455.55 | 538.63 | 0.097 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.78 |
| json | 574.83 |
| narrative | 311.18 |
| prose | 335.27 |
| math | 656.06 |
| reasoning | 552.08 |
| summary | 371.27 |
| format | 714.54 |
| ceiling_count | 1045.13 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
