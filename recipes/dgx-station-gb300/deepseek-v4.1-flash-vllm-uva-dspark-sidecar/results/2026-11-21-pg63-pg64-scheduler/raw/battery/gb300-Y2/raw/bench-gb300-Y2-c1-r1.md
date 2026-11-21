## gb300-Y2-c1-r1 (2026-11-21T16:43:50Z)

dsv41-5way gb300-Y2 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 443.61 | 535.42 | 0.166 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 827.67 |
| json | 521.84 |
| narrative | 310.5 |
| prose | 334.97 |
| math | 642.71 |
| reasoning | 556.87 |
| summary | 335.16 |
| format | 753.61 |
| ceiling_count | 1018.22 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
