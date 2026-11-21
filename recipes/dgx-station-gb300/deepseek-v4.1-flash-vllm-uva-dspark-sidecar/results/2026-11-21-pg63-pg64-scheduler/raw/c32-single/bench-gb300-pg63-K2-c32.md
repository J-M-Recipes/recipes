## gb300-pg63-K2-c32 (2026-11-21T17:10:28Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3562.3 | 150.5 | 0.274 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.88 |
| json | 146.09 |
| narrative | 83.61 |
| prose | 115.7 |
| math | 199.11 |
| reasoning | 154.45 |
| summary | 103.79 |
| format | 196.37 |
| ceiling_count | 217.15 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
