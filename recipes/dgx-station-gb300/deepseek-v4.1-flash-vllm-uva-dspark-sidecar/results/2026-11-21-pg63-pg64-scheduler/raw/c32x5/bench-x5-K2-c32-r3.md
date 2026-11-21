## x5-K2-c32-r3 (2026-11-21T17:11:52Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3683.14 | 150.39 | 0.185 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 203.06 |
| json | 143.49 |
| narrative | 82.2 |
| prose | 118.32 |
| math | 196.29 |
| reasoning | 153.1 |
| summary | 110.29 |
| format | 196.39 |
| ceiling_count | 226.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
