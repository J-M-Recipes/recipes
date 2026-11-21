## diag-pg62-c32-r3 (2026-11-21T13:35:49Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3625.99 | 148.31 | 0.197 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 201.62 |
| json | 140.34 |
| narrative | 80.16 |
| prose | 119.12 |
| math | 190.42 |
| reasoning | 152.01 |
| summary | 107.29 |
| format | 195.5 |
| ceiling_count | 209.46 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
