## gb300-pg62-A3-c1-r3 (2026-11-21T12:40:38Z)

dsv41-5way gb300-pg62-A3 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 478.31 | 541.93 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 795.03 |
| json | 565.35 |
| narrative | 312.41 |
| prose | 335.05 |
| math | 640.5 |
| reasoning | 559.07 |
| summary | 327.71 |
| format | 800.34 |
| ceiling_count | 1015.47 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
