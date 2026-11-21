## x4-A7-c32-r5 (2026-11-21T15:03:25Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3608.28 | 149.71 | 0.216 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.92 |
| json | 144.88 |
| narrative | 80.99 |
| prose | 117.86 |
| math | 189.72 |
| reasoning | 152.87 |
| summary | 106.68 |
| format | 199.77 |
| ceiling_count | 213.88 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
