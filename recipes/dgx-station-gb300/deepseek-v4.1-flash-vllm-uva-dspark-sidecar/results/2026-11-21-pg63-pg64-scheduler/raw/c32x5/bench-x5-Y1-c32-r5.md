## x5-Y1-c32-r5 (2026-11-21T16:31:21Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3685.64 | 149.26 | 0.182 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 201.51 |
| json | 147.63 |
| narrative | 81.75 |
| prose | 117.52 |
| math | 189.13 |
| reasoning | 152.62 |
| summary | 107.27 |
| format | 196.67 |
| ceiling_count | 209.53 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
