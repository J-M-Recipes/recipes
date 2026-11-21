## gb300-pg62-A1-c1-r3 (2026-11-21T10:51:30Z)

dsv41-5way gb300-pg62-A1 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 478.48 | 547.32 | 0.036 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 795.23 |
| json | 529.2 |
| narrative | 310.07 |
| prose | 335.88 |
| math | 700.44 |
| reasoning | 556.51 |
| summary | 353.52 |
| format | 797.7 |
| ceiling_count | 1051.75 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
