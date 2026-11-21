## diag-A5-cold-r3 (2026-11-21T14:03:10Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3488.19 | 150.01 | 0.239 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.18 |
| json | 141.62 |
| narrative | 82.7 |
| prose | 115.58 |
| math | 197.97 |
| reasoning | 150.36 |
| summary | 108.8 |
| format | 196.84 |
| ceiling_count | 219.6 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
