## gb300-E1-c1-r3 (2026-11-21T12:55:47Z)

dsv41-5way gb300-E1 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 485.97 | 558.56 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 792.74 |
| json | 584.07 |
| narrative | 311.77 |
| prose | 333.65 |
| math | 678.88 |
| reasoning | 556.66 |
| summary | 315.45 |
| format | 895.29 |
| ceiling_count | 1041.89 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
