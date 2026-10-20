## B1a-c1 (2026-10-20T02:38:03Z)

Lane B R2 B1a (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 79.99 | 95.02 | 0.294 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 129.16 |
| json | 99.94 |
| narrative | 58.5 |
| prose | 71.37 |
| math | 121.37 |
| reasoning | 86.53 |
| summary | 68.23 |
| format | 125.02 |
| ceiling_count | 144.38 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
