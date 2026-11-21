## x5-K1-c32-r4 (2026-11-21T16:10:09Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3651.49 | 150.5 | 0.195 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 202.16 |
| json | 146.73 |
| narrative | 82.15 |
| prose | 116.93 |
| math | 200.37 |
| reasoning | 150.64 |
| summary | 108.64 |
| format | 196.36 |
| ceiling_count | 221.61 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
