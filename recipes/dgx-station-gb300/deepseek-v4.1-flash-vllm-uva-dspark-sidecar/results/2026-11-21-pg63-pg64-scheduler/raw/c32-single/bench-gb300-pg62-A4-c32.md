## gb300-pg62-A4-c32 (2026-11-21T13:32:34Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3596.69 | 149.52 | 0.224 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.81 |
| json | 145.06 |
| narrative | 81.89 |
| prose | 116.73 |
| math | 191.05 |
| reasoning | 155.38 |
| summary | 106.78 |
| format | 193.49 |
| ceiling_count | 214.86 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
