## gb300-E2-c16-r3 (2026-11-21T13:11:11Z)

dsv41-5way gb300-E2 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2488.07 | 210.19 | 0.171 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 296.05 |
| json | 193.47 |
| narrative | 121.56 |
| prose | 162.36 |
| math | 266.6 |
| reasoning | 215.63 |
| summary | 155.57 |
| format | 270.26 |
| ceiling_count | 312.93 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
