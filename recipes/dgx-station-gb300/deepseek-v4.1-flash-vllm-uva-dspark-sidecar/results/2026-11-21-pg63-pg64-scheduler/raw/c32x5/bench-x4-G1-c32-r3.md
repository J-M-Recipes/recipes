## x4-G1-c32-r3 (2026-11-21T14:30:48Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3532.51 | 149.36 | 0.244 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 208.25 |
| json | 139.91 |
| narrative | 81.48 |
| prose | 118.25 |
| math | 190.67 |
| reasoning | 150.38 |
| summary | 109.34 |
| format | 196.58 |
| ceiling_count | 215.89 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
