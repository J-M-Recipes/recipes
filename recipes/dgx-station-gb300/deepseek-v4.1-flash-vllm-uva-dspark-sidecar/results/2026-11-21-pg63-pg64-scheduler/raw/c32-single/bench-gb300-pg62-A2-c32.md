## gb300-pg62-A2-c32 (2026-11-21T11:59:49Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3643.59 | 147.67 | 0.207 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 201.37 |
| json | 136.02 |
| narrative | 81.3 |
| prose | 114.56 |
| math | 190.9 |
| reasoning | 153.8 |
| summary | 104.58 |
| format | 198.83 |
| ceiling_count | 210.1 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
