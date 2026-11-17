## r7-a3b-c16-r2 (2026-11-17T01:38:02Z)

dsv41-5way r7-a3b c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 545.31 | 42.58 | 0.515 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 56.13 |
| json | 45.44 |
| narrative | 23.33 |
| prose | 29.58 |
| math | 52.47 |
| reasoning | 42.01 |
| summary | 28.45 |
| format | 63.26 |
| ceiling_count | 78.62 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
