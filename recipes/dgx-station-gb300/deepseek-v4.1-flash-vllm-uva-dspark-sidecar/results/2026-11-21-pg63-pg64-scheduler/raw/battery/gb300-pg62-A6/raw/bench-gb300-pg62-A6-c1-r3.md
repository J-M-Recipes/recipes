## gb300-pg62-A6-c1-r3 (2026-11-21T14:06:14Z)

dsv41-5way gb300-pg62-A6 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 479.94 | 544.24 | 0.035 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.38 |
| json | 512.87 |
| narrative | 310.98 |
| prose | 334.82 |
| math | 673.79 |
| reasoning | 558.12 |
| summary | 362.92 |
| format | 807.0 |
| ceiling_count | 1047.47 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
