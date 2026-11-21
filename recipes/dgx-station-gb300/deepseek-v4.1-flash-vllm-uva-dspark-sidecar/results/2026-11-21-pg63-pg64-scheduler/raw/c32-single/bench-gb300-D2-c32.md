## gb300-D2-c32 (2026-11-21T12:32:25Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3408.53 | 148.66 | 0.311 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 197.61 |
| json | 143.99 |
| narrative | 83.97 |
| prose | 117.46 |
| math | 190.9 |
| reasoning | 150.11 |
| summary | 106.09 |
| format | 199.18 |
| ceiling_count | 215.45 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
