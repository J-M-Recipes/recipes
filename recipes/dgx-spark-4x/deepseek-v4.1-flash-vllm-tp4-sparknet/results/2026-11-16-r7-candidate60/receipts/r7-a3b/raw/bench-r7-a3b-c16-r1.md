## r7-a3b-c16-r1 (2026-11-17T01:37:18Z)

dsv41-5way r7-a3b c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 522.83 | 41.99 | 0.615 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 52.34 |
| json | 45.07 |
| narrative | 24.18 |
| prose | 30.96 |
| math | 50.23 |
| reasoning | 42.21 |
| summary | 28.78 |
| format | 62.16 |
| ceiling_count | 79.04 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
