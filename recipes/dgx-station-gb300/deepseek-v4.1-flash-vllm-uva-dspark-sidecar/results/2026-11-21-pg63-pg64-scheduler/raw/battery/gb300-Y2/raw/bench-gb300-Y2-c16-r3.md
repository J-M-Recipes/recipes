## gb300-Y2-c16-r3 (2026-11-21T16:44:26Z)

dsv41-5way gb300-Y2 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2482.03 | 213.19 | 0.181 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 295.12 |
| json | 225.6 |
| narrative | 117.38 |
| prose | 159.73 |
| math | 271.76 |
| reasoning | 207.99 |
| summary | 150.24 |
| format | 277.68 |
| ceiling_count | 314.05 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
