## B0-c1 (2026-10-20T00:04:25Z)

Lane B R2 B0 (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 81.57 | 94.8 | 0.239 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 129.68 |
| json | 99.45 |
| narrative | 58.19 |
| prose | 71.08 |
| math | 120.3 |
| reasoning | 86.1 |
| summary | 67.75 |
| format | 125.86 |
| ceiling_count | 145.5 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
