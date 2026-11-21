## gb300-Y2-c1-r3 (2026-11-21T16:44:00Z)

dsv41-5way gb300-Y2 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 485.54 | 556.18 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.48 |
| json | 538.72 |
| narrative | 311.8 |
| prose | 334.14 |
| math | 662.38 |
| reasoning | 557.3 |
| summary | 392.65 |
| format | 857.95 |
| ceiling_count | 1044.22 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
