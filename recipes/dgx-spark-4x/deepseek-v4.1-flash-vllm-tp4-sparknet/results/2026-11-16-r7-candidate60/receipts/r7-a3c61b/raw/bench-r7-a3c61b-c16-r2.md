## r7-a3c61b-c16-r2 (2026-11-17T02:08:21Z)

dsv41-5way r7-a3c61b c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 524.09 | 41.21 | 0.533 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 54.05 |
| json | 43.15 |
| narrative | 23.02 |
| prose | 30.3 |
| math | 48.74 |
| reasoning | 39.76 |
| summary | 28.09 |
| format | 62.54 |
| ceiling_count | 78.84 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
