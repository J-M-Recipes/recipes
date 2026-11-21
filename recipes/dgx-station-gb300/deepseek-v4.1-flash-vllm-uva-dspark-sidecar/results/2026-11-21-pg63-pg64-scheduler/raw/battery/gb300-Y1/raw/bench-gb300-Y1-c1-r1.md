## gb300-Y1-c1-r1 (2026-11-21T16:22:33Z)

dsv41-5way gb300-Y1 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 455.9 | 557.17 | 0.173 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.34 |
| json | 598.22 |
| narrative | 310.4 |
| prose | 335.19 |
| math | 644.23 |
| reasoning | 570.25 |
| summary | 346.46 |
| format | 826.27 |
| ceiling_count | 1024.63 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
