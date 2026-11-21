## gb300-B-c16-r2 (2026-11-21T11:07:20Z)

dsv41-5way gb300-B c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2403.07 | 207.83 | 0.236 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.66 |
| json | 185.04 |
| narrative | 126.61 |
| prose | 161.77 |
| math | 273.84 |
| reasoning | 211.9 |
| summary | 141.4 |
| format | 268.42 |
| ceiling_count | 306.99 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
