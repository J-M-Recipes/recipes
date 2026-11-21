## gb300-G1-c1-r1 (2026-11-21T14:22:25Z)

dsv41-5way gb300-G1 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 456.92 | 556.79 | 0.17 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 827.04 |
| json | 563.6 |
| narrative | 310.81 |
| prose | 335.02 |
| math | 658.36 |
| reasoning | 553.69 |
| summary | 321.54 |
| format | 884.25 |
| ceiling_count | 1053.5 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
