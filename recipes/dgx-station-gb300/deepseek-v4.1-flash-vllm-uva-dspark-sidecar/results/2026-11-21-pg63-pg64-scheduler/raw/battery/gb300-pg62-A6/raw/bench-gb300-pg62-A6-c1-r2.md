## gb300-pg62-A6-c1-r2 (2026-11-21T14:06:10Z)

dsv41-5way gb300-pg62-A6 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 469.36 | 536.47 | 0.036 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.86 |
| json | 554.01 |
| narrative | 312.22 |
| prose | 334.76 |
| math | 660.65 |
| reasoning | 559.39 |
| summary | 321.1 |
| format | 755.77 |
| ceiling_count | 999.02 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
