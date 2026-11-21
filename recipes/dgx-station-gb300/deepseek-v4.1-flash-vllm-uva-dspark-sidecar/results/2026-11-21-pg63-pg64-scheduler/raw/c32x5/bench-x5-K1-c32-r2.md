## x5-K1-c32-r2 (2026-11-21T16:09:44Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3736.81 | 150.33 | 0.177 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 208.85 |
| json | 144.16 |
| narrative | 79.81 |
| prose | 117.2 |
| math | 192.43 |
| reasoning | 157.71 |
| summary | 105.92 |
| format | 196.58 |
| ceiling_count | 215.36 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
