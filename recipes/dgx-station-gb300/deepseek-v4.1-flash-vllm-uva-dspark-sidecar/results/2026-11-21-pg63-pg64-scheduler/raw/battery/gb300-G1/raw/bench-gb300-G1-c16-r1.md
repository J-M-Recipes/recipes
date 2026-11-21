## gb300-G1-c16-r1 (2026-11-21T14:22:39Z)

dsv41-5way gb300-G1 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1915.57 | 208.78 | 0.568 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 284.38 |
| json | 205.15 |
| narrative | 122.47 |
| prose | 159.31 |
| math | 274.36 |
| reasoning | 212.68 |
| summary | 147.95 |
| format | 263.93 |
| ceiling_count | 308.58 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
