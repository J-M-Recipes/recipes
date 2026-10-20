## B2-c1 (2026-10-20T03:27:37Z)

Lane B R2 B2 (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 78.72 | 94.52 | 0.327 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 124.25 |
| json | 99.79 |
| narrative | 58.81 |
| prose | 71.47 |
| math | 122.35 |
| reasoning | 86.04 |
| summary | 68.31 |
| format | 125.17 |
| ceiling_count | 146.39 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
