## gb300-pg62-A6-c32 (2026-11-21T14:12:58Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3701.07 | 150.34 | 0.187 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.54 |
| json | 141.61 |
| narrative | 80.65 |
| prose | 115.68 |
| math | 195.72 |
| reasoning | 154.54 |
| summary | 106.32 |
| format | 202.7 |
| ceiling_count | 214.21 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
