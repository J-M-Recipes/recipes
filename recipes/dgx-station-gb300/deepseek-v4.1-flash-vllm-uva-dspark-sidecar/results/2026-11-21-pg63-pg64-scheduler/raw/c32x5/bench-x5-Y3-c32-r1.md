## x5-Y3-c32-r1 (2026-11-21T17:48:58Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3647.79 | 150.31 | 0.209 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 208.9 |
| json | 146.44 |
| narrative | 83.05 |
| prose | 118.39 |
| math | 184.95 |
| reasoning | 156.77 |
| summary | 107.18 |
| format | 196.77 |
| ceiling_count | 216.05 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
