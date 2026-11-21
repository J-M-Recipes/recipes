## gb300-E2-c16-r2 (2026-11-21T13:11:01Z)

dsv41-5way gb300-E2 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2406.52 | 208.78 | 0.232 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 284.38 |
| json | 194.32 |
| narrative | 121.86 |
| prose | 166.97 |
| math | 265.35 |
| reasoning | 218.54 |
| summary | 153.48 |
| format | 265.31 |
| ceiling_count | 309.96 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
