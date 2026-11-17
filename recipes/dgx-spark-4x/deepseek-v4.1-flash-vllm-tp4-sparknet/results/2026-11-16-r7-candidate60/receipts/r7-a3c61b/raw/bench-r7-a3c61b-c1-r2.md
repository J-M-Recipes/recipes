## r7-a3c61b-c1-r2 (2026-11-17T02:07:07Z)

dsv41-5way r7-a3c61b c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 103.99 | 121.8 | 0.18 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 165.69 |
| json | 148.1 |
| narrative | 70.67 |
| prose | 85.65 |
| math | 136.93 |
| reasoning | 118.57 |
| summary | 88.63 |
| format | 160.13 |
| ceiling_count | 189.01 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
