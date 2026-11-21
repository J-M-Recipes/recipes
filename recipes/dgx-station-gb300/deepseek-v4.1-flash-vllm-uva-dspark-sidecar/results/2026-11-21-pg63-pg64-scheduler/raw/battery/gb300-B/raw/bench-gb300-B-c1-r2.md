## gb300-B-c1-r2 (2026-11-21T11:07:00Z)

dsv41-5way gb300-B c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 467.88 | 559.46 | 0.099 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.92 |
| json | 590.29 |
| narrative | 311.33 |
| prose | 334.93 |
| math | 680.75 |
| reasoning | 558.95 |
| summary | 356.67 |
| format | 847.86 |
| ceiling_count | 1017.58 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
