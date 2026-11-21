## gb300-pg62-A1-c1-r1 (2026-11-21T10:51:22Z)

dsv41-5way gb300-pg62-A1 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 464.13 | 544.31 | 0.055 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 796.09 |
| json | 563.32 |
| narrative | 309.95 |
| prose | 371.76 |
| math | 625.63 |
| reasoning | 559.71 |
| summary | 340.74 |
| format | 787.3 |
| ceiling_count | 1051.51 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
