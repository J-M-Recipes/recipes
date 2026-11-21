## gb300-pg62-A2-c1-r1 (2026-11-21T11:52:53Z)

dsv41-5way gb300-pg62-A2 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 446.48 | 533.23 | 0.163 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.06 |
| json | 532.31 |
| narrative | 311.1 |
| prose | 335.01 |
| math | 612.29 |
| reasoning | 552.76 |
| summary | 345.21 |
| format | 751.1 |
| ceiling_count | 1016.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
