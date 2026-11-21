## gb300-D1-c16-r2 (2026-11-21T12:10:49Z)

dsv41-5way gb300-D1 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2404.49 | 209.55 | 0.239 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 292.04 |
| json | 192.55 |
| narrative | 121.87 |
| prose | 164.38 |
| math | 270.21 |
| reasoning | 209.6 |
| summary | 151.22 |
| format | 274.56 |
| ceiling_count | 311.21 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
