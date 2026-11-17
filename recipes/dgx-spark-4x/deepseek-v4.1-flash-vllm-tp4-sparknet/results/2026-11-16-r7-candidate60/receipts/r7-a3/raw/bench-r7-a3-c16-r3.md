## r7-a3-c16-r3 (2026-11-17T02:08:25Z)

dsv41-5way r7-a3 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 545.67 | 42.75 | 0.555 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 56.8 |
| json | 46.15 |
| narrative | 22.91 |
| prose | 30.84 |
| math | 54.08 |
| reasoning | 40.21 |
| summary | 28.79 |
| format | 62.2 |
| ceiling_count | 79.63 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
