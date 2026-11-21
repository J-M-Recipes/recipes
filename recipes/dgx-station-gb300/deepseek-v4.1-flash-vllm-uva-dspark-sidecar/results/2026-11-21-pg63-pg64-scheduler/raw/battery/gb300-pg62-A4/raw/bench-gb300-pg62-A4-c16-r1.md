## gb300-pg62-A4-c16-r1 (2026-11-21T13:25:50Z)

dsv41-5way gb300-pg62-A4 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1907.24 | 209.98 | 0.564 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.06 |
| json | 197.02 |
| narrative | 122.81 |
| prose | 158.27 |
| math | 277.73 |
| reasoning | 211.93 |
| summary | 147.44 |
| format | 271.58 |
| ceiling_count | 309.51 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
