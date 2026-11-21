## x4-A6-c32-r2 (2026-11-21T14:14:10Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3733.42 | 150.99 | 0.18 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 209.97 |
| json | 144.28 |
| narrative | 82.31 |
| prose | 115.76 |
| math | 192.89 |
| reasoning | 153.9 |
| summary | 109.53 |
| format | 199.26 |
| ceiling_count | 222.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
