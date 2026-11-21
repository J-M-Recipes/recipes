## x4-G1-c32-r4 (2026-11-21T14:31:01Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3696.96 | 150.82 | 0.185 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 207.7 |
| json | 147.13 |
| narrative | 82.86 |
| prose | 118.1 |
| math | 195.2 |
| reasoning | 151.23 |
| summary | 105.75 |
| format | 198.56 |
| ceiling_count | 217.32 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
