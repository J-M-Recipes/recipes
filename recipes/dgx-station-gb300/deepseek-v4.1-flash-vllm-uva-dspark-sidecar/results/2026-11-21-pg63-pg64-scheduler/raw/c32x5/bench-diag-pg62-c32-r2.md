## diag-pg62-c32-r2 (2026-11-21T13:35:35Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3455.86 | 150.14 | 0.272 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 200.46 |
| json | 147.54 |
| narrative | 79.77 |
| prose | 118.26 |
| math | 193.04 |
| reasoning | 153.95 |
| summary | 113.38 |
| format | 194.73 |
| ceiling_count | 216.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
