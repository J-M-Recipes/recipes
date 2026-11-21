## gb300-G2-c16-r3 (2026-11-21T14:39:07Z)

dsv41-5way gb300-G2 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2505.0 | 211.97 | 0.17 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 296.32 |
| json | 196.61 |
| narrative | 120.7 |
| prose | 162.71 |
| math | 281.3 |
| reasoning | 214.63 |
| summary | 151.53 |
| format | 271.97 |
| ceiling_count | 309.63 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
