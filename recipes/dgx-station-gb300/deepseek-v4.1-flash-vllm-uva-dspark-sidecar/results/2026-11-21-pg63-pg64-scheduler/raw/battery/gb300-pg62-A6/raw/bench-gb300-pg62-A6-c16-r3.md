## gb300-pg62-A6-c16-r3 (2026-11-21T14:06:37Z)

dsv41-5way gb300-pg62-A6 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2421.46 | 206.64 | 0.179 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.85 |
| json | 190.52 |
| narrative | 119.94 |
| prose | 161.62 |
| math | 260.03 |
| reasoning | 205.15 |
| summary | 151.91 |
| format | 270.09 |
| ceiling_count | 309.28 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
