## gb300-D1-c16-r1 (2026-11-21T12:10:37Z)

dsv41-5way gb300-D1 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1883.83 | 210.43 | 0.563 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 289.18 |
| json | 202.2 |
| narrative | 118.63 |
| prose | 167.61 |
| math | 266.94 |
| reasoning | 211.17 |
| summary | 147.64 |
| format | 280.08 |
| ceiling_count | 311.02 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
