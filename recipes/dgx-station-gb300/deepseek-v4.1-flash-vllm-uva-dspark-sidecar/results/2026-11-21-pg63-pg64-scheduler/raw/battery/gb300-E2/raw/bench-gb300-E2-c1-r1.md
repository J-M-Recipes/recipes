## gb300-E2-c1-r1 (2026-11-21T13:10:36Z)

dsv41-5way gb300-E2 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 451.88 | 536.65 | 0.161 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 825.52 |
| json | 551.8 |
| narrative | 310.47 |
| prose | 334.84 |
| math | 646.8 |
| reasoning | 553.19 |
| summary | 312.07 |
| format | 758.48 |
| ceiling_count | 1016.44 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
