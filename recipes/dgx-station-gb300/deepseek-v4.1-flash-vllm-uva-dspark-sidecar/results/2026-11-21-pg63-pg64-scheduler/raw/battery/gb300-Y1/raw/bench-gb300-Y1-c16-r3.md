## gb300-Y1-c16-r3 (2026-11-21T16:23:10Z)

dsv41-5way gb300-Y1 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2482.98 | 212.16 | 0.178 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 296.4 |
| json | 187.71 |
| narrative | 119.21 |
| prose | 163.55 |
| math | 279.47 |
| reasoning | 213.51 |
| summary | 153.61 |
| format | 283.82 |
| ceiling_count | 309.71 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
