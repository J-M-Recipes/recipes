## gb300-pg63-K2-c16-r1 (2026-11-21T17:03:42Z)

dsv41-5way gb300-pg63-K2 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1887.93 | 209.57 | 0.567 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 285.3 |
| json | 194.89 |
| narrative | 120.61 |
| prose | 164.27 |
| math | 271.94 |
| reasoning | 214.41 |
| summary | 147.79 |
| format | 277.35 |
| ceiling_count | 310.76 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
