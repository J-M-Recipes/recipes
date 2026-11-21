## x4-G1-c32-r2 (2026-11-21T14:30:36Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3585.21 | 149.77 | 0.225 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 203.24 |
| json | 144.69 |
| narrative | 82.36 |
| prose | 117.19 |
| math | 188.38 |
| reasoning | 157.65 |
| summary | 110.65 |
| format | 193.99 |
| ceiling_count | 217.21 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
