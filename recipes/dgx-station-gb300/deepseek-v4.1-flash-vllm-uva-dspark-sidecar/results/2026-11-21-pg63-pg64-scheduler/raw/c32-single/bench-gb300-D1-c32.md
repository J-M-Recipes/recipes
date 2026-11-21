## gb300-D1-c32 (2026-11-21T12:17:22Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3529.82 | 150.37 | 0.301 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.08 |
| json | 144.78 |
| narrative | 79.28 |
| prose | 114.5 |
| math | 196.77 |
| reasoning | 155.3 |
| summary | 108.83 |
| format | 197.39 |
| ceiling_count | 220.09 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
