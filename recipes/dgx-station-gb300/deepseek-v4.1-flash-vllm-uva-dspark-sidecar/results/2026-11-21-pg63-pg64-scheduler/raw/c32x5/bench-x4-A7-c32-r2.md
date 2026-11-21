## x4-A7-c32-r2 (2026-11-21T15:02:47Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3669.3 | 150.16 | 0.193 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 202.55 |
| json | 145.07 |
| narrative | 81.91 |
| prose | 118.19 |
| math | 194.71 |
| reasoning | 154.46 |
| summary | 106.8 |
| format | 197.63 |
| ceiling_count | 220.17 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
