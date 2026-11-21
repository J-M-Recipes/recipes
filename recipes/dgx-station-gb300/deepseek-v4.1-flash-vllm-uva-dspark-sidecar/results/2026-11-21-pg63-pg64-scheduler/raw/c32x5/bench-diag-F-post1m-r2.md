## diag-F-post1m-r2 (2026-11-21T13:55:22Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3714.96 | 150.04 | 0.188 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.01 |
| json | 142.5 |
| narrative | 83.14 |
| prose | 117.11 |
| math | 190.8 |
| reasoning | 153.0 |
| summary | 110.33 |
| format | 198.47 |
| ceiling_count | 214.6 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
