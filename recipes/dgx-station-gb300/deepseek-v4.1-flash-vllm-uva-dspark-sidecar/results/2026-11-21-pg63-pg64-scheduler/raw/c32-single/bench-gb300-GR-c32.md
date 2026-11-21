## gb300-GR-c32 (2026-11-21T11:44:10Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3627.41 | 149.87 | 0.233 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 201.3 |
| json | 142.5 |
| narrative | 82.55 |
| prose | 114.56 |
| math | 199.25 |
| reasoning | 152.55 |
| summary | 106.95 |
| format | 199.28 |
| ceiling_count | 223.08 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
