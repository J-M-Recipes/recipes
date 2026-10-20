## B0ctl-c1 (2026-10-20T01:13:07Z)

Lane B R2 B0ctl (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 82.16 | 95.04 | 0.232 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 130.49 |
| json | 99.94 |
| narrative | 58.36 |
| prose | 71.6 |
| math | 121.32 |
| reasoning | 86.01 |
| summary | 67.74 |
| format | 124.83 |
| ceiling_count | 136.98 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
