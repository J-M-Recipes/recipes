## B1b-c1 (2026-10-20T03:02:15Z)

Lane B R2 B1b (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 81.85 | 94.84 | 0.234 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 129.73 |
| json | 99.64 |
| narrative | 58.48 |
| prose | 68.99 |
| math | 120.81 |
| reasoning | 86.22 |
| summary | 68.21 |
| format | 126.63 |
| ceiling_count | 145.02 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
