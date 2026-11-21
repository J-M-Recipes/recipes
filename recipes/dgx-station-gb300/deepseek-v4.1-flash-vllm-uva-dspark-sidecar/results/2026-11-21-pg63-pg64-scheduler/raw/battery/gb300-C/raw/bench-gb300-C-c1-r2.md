## gb300-C-c1-r2 (2026-11-21T11:22:01Z)

dsv41-5way gb300-C c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 456.48 | 543.8 | 0.097 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.4 |
| json | 537.09 |
| narrative | 311.73 |
| prose | 334.82 |
| math | 616.53 |
| reasoning | 558.83 |
| summary | 347.12 |
| format | 849.87 |
| ceiling_count | 1015.52 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
