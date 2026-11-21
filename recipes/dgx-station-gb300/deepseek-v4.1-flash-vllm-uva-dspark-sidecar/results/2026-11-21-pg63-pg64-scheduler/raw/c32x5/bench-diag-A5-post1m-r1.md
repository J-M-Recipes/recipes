## diag-A5-post1m-r1 (2026-11-21T14:04:05Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3700.11 | 149.78 | 0.186 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 202.61 |
| json | 144.97 |
| narrative | 81.1 |
| prose | 116.21 |
| math | 193.12 |
| reasoning | 153.23 |
| summary | 113.65 |
| format | 193.36 |
| ceiling_count | 213.33 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
