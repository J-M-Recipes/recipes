## x5-K1-c32-r5 (2026-11-21T16:10:21Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3599.35 | 149.98 | 0.21 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.23 |
| json | 144.73 |
| narrative | 82.58 |
| prose | 118.08 |
| math | 194.25 |
| reasoning | 153.46 |
| summary | 109.69 |
| format | 190.8 |
| ceiling_count | 213.42 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
