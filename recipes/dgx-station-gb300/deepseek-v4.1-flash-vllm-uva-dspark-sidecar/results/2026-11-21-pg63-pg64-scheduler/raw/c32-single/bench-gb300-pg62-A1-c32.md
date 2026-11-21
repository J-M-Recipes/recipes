## gb300-pg62-A1-c32 (2026-11-21T10:58:15Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3684.06 | 150.52 | 0.208 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 208.73 |
| json | 146.68 |
| narrative | 82.23 |
| prose | 114.43 |
| math | 189.65 |
| reasoning | 155.81 |
| summary | 107.09 |
| format | 199.57 |
| ceiling_count | 217.0 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
