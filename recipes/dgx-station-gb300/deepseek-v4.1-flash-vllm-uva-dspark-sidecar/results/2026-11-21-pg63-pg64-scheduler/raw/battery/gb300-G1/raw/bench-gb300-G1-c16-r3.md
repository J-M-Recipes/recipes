## gb300-G1-c16-r3 (2026-11-21T14:23:02Z)

dsv41-5way gb300-G1 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2479.14 | 208.92 | 0.176 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 290.21 |
| json | 196.88 |
| narrative | 118.51 |
| prose | 160.82 |
| math | 275.12 |
| reasoning | 214.84 |
| summary | 143.16 |
| format | 271.82 |
| ceiling_count | 303.79 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
