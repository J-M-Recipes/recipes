## diag-A5-cold-r2 (2026-11-21T14:02:58Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3659.37 | 150.23 | 0.192 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.75 |
| json | 143.99 |
| narrative | 80.06 |
| prose | 117.87 |
| math | 191.46 |
| reasoning | 152.12 |
| summary | 107.83 |
| format | 201.76 |
| ceiling_count | 215.84 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
