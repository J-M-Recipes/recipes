## gb300-GR-c1-r1 (2026-11-21T11:37:16Z)

dsv41-5way gb300-GR c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 461.44 | 552.58 | 0.164 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.22 |
| json | 531.63 |
| narrative | 310.75 |
| prose | 335.01 |
| math | 628.98 |
| reasoning | 570.54 |
| summary | 359.9 |
| format | 857.63 |
| ceiling_count | 1019.7 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
