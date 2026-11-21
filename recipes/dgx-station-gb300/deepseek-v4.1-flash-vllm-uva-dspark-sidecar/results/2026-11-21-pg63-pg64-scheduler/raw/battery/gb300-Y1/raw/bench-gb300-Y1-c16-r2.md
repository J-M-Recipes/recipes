## gb300-Y1-c16-r2 (2026-11-21T16:23:00Z)

dsv41-5way gb300-Y1 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2393.55 | 208.78 | 0.239 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 295.43 |
| json | 194.43 |
| narrative | 118.99 |
| prose | 161.62 |
| math | 265.47 |
| reasoning | 203.9 |
| summary | 151.01 |
| format | 279.35 |
| ceiling_count | 310.06 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
