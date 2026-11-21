## diag-pg62-c32-r1 (2026-11-21T13:35:22Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3546.65 | 149.51 | 0.233 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.46 |
| json | 141.49 |
| narrative | 81.43 |
| prose | 116.38 |
| math | 188.58 |
| reasoning | 156.74 |
| summary | 112.55 |
| format | 192.43 |
| ceiling_count | 209.74 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
