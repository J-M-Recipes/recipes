## r7-a2-c16-r2 (2026-11-17T01:46:52Z)

dsv41-5way r7-a2 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 555.78 | 42.89 | 0.51 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 58.8 |
| json | 44.55 |
| narrative | 23.12 |
| prose | 30.15 |
| math | 51.16 |
| reasoning | 41.75 |
| summary | 28.92 |
| format | 64.7 |
| ceiling_count | 79.24 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
