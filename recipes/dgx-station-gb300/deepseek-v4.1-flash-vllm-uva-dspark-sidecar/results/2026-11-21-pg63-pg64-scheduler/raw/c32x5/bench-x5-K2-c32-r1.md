## x5-K2-c32-r1 (2026-11-21T17:11:27Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3745.54 | 151.07 | 0.167 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.7 |
| json | 146.55 |
| narrative | 82.58 |
| prose | 119.54 |
| math | 191.89 |
| reasoning | 152.0 |
| summary | 109.64 |
| format | 199.66 |
| ceiling_count | 215.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
