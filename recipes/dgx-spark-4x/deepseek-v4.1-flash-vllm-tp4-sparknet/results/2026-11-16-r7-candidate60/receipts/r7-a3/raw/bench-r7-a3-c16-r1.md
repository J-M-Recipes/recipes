## r7-a3-c16-r1 (2026-11-17T02:07:00Z)

dsv41-5way r7-a3 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 556.48 | 43.49 | 0.518 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 58.61 |
| json | 45.57 |
| narrative | 24.06 |
| prose | 31.25 |
| math | 53.51 |
| reasoning | 42.32 |
| summary | 29.09 |
| format | 63.53 |
| ceiling_count | 79.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
