## gb300-G2-c16-r2 (2026-11-21T14:38:57Z)

dsv41-5way gb300-G2 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2397.11 | 208.97 | 0.231 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 295.73 |
| json | 190.61 |
| narrative | 122.58 |
| prose | 160.46 |
| math | 264.75 |
| reasoning | 207.88 |
| summary | 147.64 |
| format | 282.1 |
| ceiling_count | 311.87 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
