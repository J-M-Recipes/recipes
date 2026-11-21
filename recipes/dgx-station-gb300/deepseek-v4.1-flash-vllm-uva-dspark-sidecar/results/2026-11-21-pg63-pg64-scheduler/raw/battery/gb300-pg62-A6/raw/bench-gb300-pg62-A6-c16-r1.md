## gb300-pg62-A6-c16-r1 (2026-11-21T14:06:17Z)

dsv41-5way gb300-pg62-A6 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2438.78 | 209.58 | 0.197 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.67 |
| json | 194.49 |
| narrative | 119.32 |
| prose | 164.46 |
| math | 263.86 |
| reasoning | 216.55 |
| summary | 151.54 |
| format | 272.72 |
| ceiling_count | 310.43 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
