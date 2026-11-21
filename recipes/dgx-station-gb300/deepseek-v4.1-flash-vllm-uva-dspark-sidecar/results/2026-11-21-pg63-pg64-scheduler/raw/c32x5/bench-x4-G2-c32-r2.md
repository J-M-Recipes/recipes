## x4-G2-c32-r2 (2026-11-21T14:46:42Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3758.71 | 150.86 | 0.167 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.34 |
| json | 137.36 |
| narrative | 80.77 |
| prose | 119.0 |
| math | 196.53 |
| reasoning | 155.34 |
| summary | 110.92 |
| format | 200.59 |
| ceiling_count | 215.86 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
