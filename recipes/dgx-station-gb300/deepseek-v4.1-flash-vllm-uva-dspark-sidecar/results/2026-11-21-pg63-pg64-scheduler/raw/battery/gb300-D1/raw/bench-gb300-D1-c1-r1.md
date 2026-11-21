## gb300-D1-c1-r1 (2026-11-21T12:10:23Z)

dsv41-5way gb300-D1 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 463.5 | 565.72 | 0.168 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.17 |
| json | 538.24 |
| narrative | 311.27 |
| prose | 335.05 |
| math | 696.92 |
| reasoning | 570.31 |
| summary | 369.29 |
| format | 878.53 |
| ceiling_count | 1023.34 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
