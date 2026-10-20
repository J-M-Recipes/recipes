## B3-c1 (2026-10-20T03:46:51Z)

Lane B R2 B3 (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 82.19 | 95.25 | 0.242 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 127.48 |
| json | 100.22 |
| narrative | 59.05 |
| prose | 71.73 |
| math | 123.21 |
| reasoning | 86.55 |
| summary | 68.15 |
| format | 125.57 |
| ceiling_count | 146.36 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
