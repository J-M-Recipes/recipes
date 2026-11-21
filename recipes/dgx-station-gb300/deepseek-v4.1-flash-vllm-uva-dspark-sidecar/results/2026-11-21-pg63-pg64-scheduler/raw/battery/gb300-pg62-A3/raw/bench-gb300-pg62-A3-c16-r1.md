## gb300-pg62-A3-c16-r1 (2026-11-21T12:40:42Z)

dsv41-5way gb300-pg62-A3 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1890.02 | 209.35 | 0.562 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 296.73 |
| json | 189.23 |
| narrative | 122.74 |
| prose | 162.34 |
| math | 272.09 |
| reasoning | 205.37 |
| summary | 147.47 |
| format | 278.85 |
| ceiling_count | 310.77 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
