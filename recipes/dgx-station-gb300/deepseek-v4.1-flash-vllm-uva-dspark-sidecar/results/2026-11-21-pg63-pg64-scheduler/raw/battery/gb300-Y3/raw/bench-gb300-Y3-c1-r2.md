## gb300-Y3-c1-r2 (2026-11-21T17:41:06Z)

dsv41-5way gb300-Y3 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 466.77 | 568.02 | 0.1 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.51 |
| json | 652.35 |
| narrative | 311.7 |
| prose | 334.88 |
| math | 682.62 |
| reasoning | 558.97 |
| summary | 379.18 |
| format | 829.99 |
| ceiling_count | 1048.69 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
