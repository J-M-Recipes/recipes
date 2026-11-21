## gb300-D2-c1-r1 (2026-11-21T12:25:27Z)

dsv41-5way gb300-D2 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 451.14 | 552.19 | 0.168 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 824.3 |
| json | 515.96 |
| narrative | 311.09 |
| prose | 334.97 |
| math | 614.85 |
| reasoning | 574.56 |
| summary | 351.81 |
| format | 889.97 |
| ceiling_count | 1022.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
