## gb300-D2-c1-r3 (2026-11-21T12:25:37Z)

dsv41-5way gb300-D2 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 480.38 | 547.23 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 795.5 |
| json | 524.18 |
| narrative | 311.82 |
| prose | 334.19 |
| math | 681.05 |
| reasoning | 558.35 |
| summary | 349.79 |
| format | 822.92 |
| ceiling_count | 1049.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
