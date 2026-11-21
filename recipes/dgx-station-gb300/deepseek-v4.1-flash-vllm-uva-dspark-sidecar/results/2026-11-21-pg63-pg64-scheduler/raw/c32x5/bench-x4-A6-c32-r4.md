## x4-A6-c32-r4 (2026-11-21T14:14:35Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3777.08 | 151.24 | 0.174 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 207.01 |
| json | 144.01 |
| narrative | 80.36 |
| prose | 120.23 |
| math | 195.05 |
| reasoning | 152.49 |
| summary | 108.5 |
| format | 202.23 |
| ceiling_count | 216.54 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
