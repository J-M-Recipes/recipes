## gb300-pg62-A2-c16-r2 (2026-11-21T11:53:19Z)

dsv41-5way gb300-pg62-A2 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2379.39 | 209.68 | 0.233 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 294.67 |
| json | 190.45 |
| narrative | 116.43 |
| prose | 163.56 |
| math | 288.21 |
| reasoning | 208.54 |
| summary | 149.68 |
| format | 265.93 |
| ceiling_count | 313.6 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
