## x5-Y2-c32-r5 (2026-11-21T16:52:38Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3704.55 | 151.07 | 0.18 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.3 |
| json | 146.77 |
| narrative | 81.52 |
| prose | 115.83 |
| math | 188.57 |
| reasoning | 156.65 |
| summary | 112.99 |
| format | 199.96 |
| ceiling_count | 217.84 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
