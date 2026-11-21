## x4-G2-c32-r5 (2026-11-21T14:47:20Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3649.77 | 149.67 | 0.206 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 207.79 |
| json | 143.37 |
| narrative | 83.96 |
| prose | 113.71 |
| math | 192.96 |
| reasoning | 152.86 |
| summary | 106.6 |
| format | 196.12 |
| ceiling_count | 206.04 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
