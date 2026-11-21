## gb300-E1-c32 (2026-11-21T13:02:33Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3515.74 | 150.05 | 0.288 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.54 |
| json | 144.79 |
| narrative | 81.42 |
| prose | 115.85 |
| math | 196.99 |
| reasoning | 154.52 |
| summary | 104.35 |
| format | 197.94 |
| ceiling_count | 224.14 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
