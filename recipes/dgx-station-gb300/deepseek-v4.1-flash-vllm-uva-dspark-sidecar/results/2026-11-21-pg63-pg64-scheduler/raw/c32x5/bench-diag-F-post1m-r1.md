## diag-F-post1m-r1 (2026-11-21T13:55:09Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3675.3 | 150.31 | 0.191 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.58 |
| json | 143.87 |
| narrative | 82.08 |
| prose | 114.72 |
| math | 193.55 |
| reasoning | 150.0 |
| summary | 112.54 |
| format | 199.14 |
| ceiling_count | 216.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
