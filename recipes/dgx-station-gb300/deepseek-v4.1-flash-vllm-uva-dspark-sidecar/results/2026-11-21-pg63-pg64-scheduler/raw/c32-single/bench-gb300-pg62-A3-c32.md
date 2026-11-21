## gb300-pg62-A3-c32 (2026-11-21T12:47:25Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3722.23 | 149.79 | 0.214 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.83 |
| json | 150.09 |
| narrative | 80.5 |
| prose | 115.53 |
| math | 189.15 |
| reasoning | 151.07 |
| summary | 107.1 |
| format | 200.07 |
| ceiling_count | 222.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
