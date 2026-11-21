## gb300-Y3-c32 (2026-11-21T17:47:58Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3358.97 | 146.65 | 0.292 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 198.53 |
| json | 137.91 |
| narrative | 81.87 |
| prose | 116.74 |
| math | 189.43 |
| reasoning | 150.62 |
| summary | 103.31 |
| format | 194.75 |
| ceiling_count | 215.38 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
