## gb300-pg62-A4-c16-r2 (2026-11-21T13:26:02Z)

dsv41-5way gb300-pg62-A4 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2374.74 | 210.89 | 0.312 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.3 |
| json | 200.84 |
| narrative | 120.6 |
| prose | 164.64 |
| math | 273.13 |
| reasoning | 211.91 |
| summary | 148.3 |
| format | 274.39 |
| ceiling_count | 308.04 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
