## gb300-Y3-c16-r3 (2026-11-21T17:41:36Z)

dsv41-5way gb300-Y3 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2462.21 | 211.45 | 0.178 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 292.92 |
| json | 186.75 |
| narrative | 120.75 |
| prose | 166.07 |
| math | 280.95 |
| reasoning | 215.19 |
| summary | 154.12 |
| format | 274.84 |
| ceiling_count | 311.38 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
