## gb300-pg62-A4-c1-r1 (2026-11-21T13:25:36Z)

dsv41-5way gb300-pg62-A4 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 453.48 | 574.25 | 0.174 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 974.01 |
| json | 564.86 |
| narrative | 310.73 |
| prose | 333.09 |
| math | 640.79 |
| reasoning | 570.79 |
| summary | 334.29 |
| format | 865.43 |
| ceiling_count | 1030.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
