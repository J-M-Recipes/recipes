## x5-Y1-c32-r2 (2026-11-21T16:30:43Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3658.66 | 149.32 | 0.193 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 201.42 |
| json | 145.44 |
| narrative | 81.37 |
| prose | 119.93 |
| math | 192.0 |
| reasoning | 150.77 |
| summary | 105.31 |
| format | 198.32 |
| ceiling_count | 216.29 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
