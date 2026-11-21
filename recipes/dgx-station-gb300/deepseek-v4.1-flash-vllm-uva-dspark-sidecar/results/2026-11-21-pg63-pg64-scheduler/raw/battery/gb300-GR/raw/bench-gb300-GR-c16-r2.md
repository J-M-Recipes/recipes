## gb300-GR-c16-r2 (2026-11-21T11:37:42Z)

dsv41-5way gb300-GR c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2421.88 | 211.12 | 0.226 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 285.66 |
| json | 193.11 |
| narrative | 122.42 |
| prose | 167.79 |
| math | 273.16 |
| reasoning | 209.35 |
| summary | 160.75 |
| format | 276.68 |
| ceiling_count | 309.19 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
