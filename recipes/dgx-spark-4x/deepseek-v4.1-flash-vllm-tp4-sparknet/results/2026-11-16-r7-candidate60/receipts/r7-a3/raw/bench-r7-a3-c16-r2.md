## r7-a3-c16-r2 (2026-11-17T02:07:42Z)

dsv41-5way r7-a3 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 535.74 | 41.78 | 0.517 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 55.13 |
| json | 44.52 |
| narrative | 22.8 |
| prose | 29.92 |
| math | 51.54 |
| reasoning | 41.56 |
| summary | 28.4 |
| format | 60.4 |
| ceiling_count | 79.36 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
