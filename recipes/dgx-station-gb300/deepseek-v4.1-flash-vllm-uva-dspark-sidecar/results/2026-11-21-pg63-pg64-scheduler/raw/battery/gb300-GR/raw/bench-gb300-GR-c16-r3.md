## gb300-GR-c16-r3 (2026-11-21T11:37:52Z)

dsv41-5way gb300-GR c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2498.51 | 211.03 | 0.168 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.49 |
| json | 196.57 |
| narrative | 119.38 |
| prose | 168.39 |
| math | 269.51 |
| reasoning | 215.24 |
| summary | 151.18 |
| format | 274.48 |
| ceiling_count | 311.77 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
