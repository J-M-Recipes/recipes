## gb300-pg62-A1-c16-r2 (2026-11-21T10:51:44Z)

dsv41-5way gb300-pg62-A1 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2453.82 | 208.26 | 0.185 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 278.54 |
| json | 188.6 |
| narrative | 120.91 |
| prose | 164.48 |
| math | 275.93 |
| reasoning | 210.96 |
| summary | 150.51 |
| format | 276.15 |
| ceiling_count | 310.56 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
