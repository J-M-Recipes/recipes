## gb300-pg62-A2-c1-r2 (2026-11-21T11:52:58Z)

dsv41-5way gb300-pg62-A2 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 457.51 | 540.42 | 0.095 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.32 |
| json | 618.84 |
| narrative | 311.44 |
| prose | 335.5 |
| math | 608.58 |
| reasoning | 558.91 |
| summary | 350.82 |
| format | 745.96 |
| ceiling_count | 1015.25 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
