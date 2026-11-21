## x5-Y1-c32-r4 (2026-11-21T16:31:08Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3724.07 | 149.75 | 0.17 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.94 |
| json | 144.57 |
| narrative | 80.57 |
| prose | 116.92 |
| math | 189.14 |
| reasoning | 156.59 |
| summary | 106.71 |
| format | 196.57 |
| ceiling_count | 211.52 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
