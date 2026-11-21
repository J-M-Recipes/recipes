## x5-K2-c32-r5 (2026-11-21T17:12:17Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3739.57 | 149.94 | 0.171 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 202.9 |
| json | 143.05 |
| narrative | 80.92 |
| prose | 117.74 |
| math | 194.14 |
| reasoning | 151.23 |
| summary | 108.18 |
| format | 201.35 |
| ceiling_count | 217.85 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
