## gb300-D2-c16-r2 (2026-11-21T12:25:53Z)

dsv41-5way gb300-D2 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2378.71 | 210.97 | 0.247 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 289.7 |
| json | 196.79 |
| narrative | 116.7 |
| prose | 164.99 |
| math | 264.49 |
| reasoning | 221.18 |
| summary | 156.94 |
| format | 276.94 |
| ceiling_count | 310.1 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
