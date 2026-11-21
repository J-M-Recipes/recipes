## x5-K2-c32-r4 (2026-11-21T17:12:05Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3759.83 | 151.28 | 0.188 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 210.74 |
| json | 143.86 |
| narrative | 82.2 |
| prose | 116.73 |
| math | 193.44 |
| reasoning | 152.85 |
| summary | 109.95 |
| format | 200.49 |
| ceiling_count | 225.3 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
