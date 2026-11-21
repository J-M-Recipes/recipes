## gb300-pg63-K1-c16-r2 (2026-11-21T16:01:59Z)

dsv41-5way gb300-pg63-K1 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2387.63 | 209.83 | 0.252 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 291.49 |
| json | 197.17 |
| narrative | 117.83 |
| prose | 166.91 |
| math | 274.02 |
| reasoning | 204.17 |
| summary | 152.03 |
| format | 274.99 |
| ceiling_count | 305.29 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
