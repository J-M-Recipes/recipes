## gb300-pg62-A7-c16-r1 (2026-11-21T14:54:50Z)

dsv41-5way gb300-pg62-A7 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1914.84 | 209.88 | 0.565 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 285.82 |
| json | 198.74 |
| narrative | 120.2 |
| prose | 163.07 |
| math | 280.46 |
| reasoning | 215.91 |
| summary | 143.58 |
| format | 271.23 |
| ceiling_count | 295.79 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
