## gb300-G1-c1-r3 (2026-11-21T14:22:35Z)

dsv41-5way gb300-G1 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 468.6 | 531.96 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 797.44 |
| json | 523.58 |
| narrative | 311.24 |
| prose | 334.76 |
| math | 613.8 |
| reasoning | 559.17 |
| summary | 361.45 |
| format | 754.26 |
| ceiling_count | 1016.65 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
