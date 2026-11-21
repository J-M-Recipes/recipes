## gb300-Y3-c1-r1 (2026-11-21T17:41:00Z)

dsv41-5way gb300-Y3 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 441.36 | 539.61 | 0.167 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 825.38 |
| json | 638.62 |
| narrative | 311.11 |
| prose | 335.15 |
| math | 611.51 |
| reasoning | 493.57 |
| summary | 364.65 |
| format | 736.87 |
| ceiling_count | 1018.78 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
