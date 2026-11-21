## gb300-Y1-c32 (2026-11-21T16:29:30Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3398.29 | 147.06 | 0.291 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 193.34 |
| json | 145.17 |
| narrative | 81.63 |
| prose | 116.03 |
| math | 194.06 |
| reasoning | 151.65 |
| summary | 100.67 |
| format | 193.89 |
| ceiling_count | 216.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
