## gb300-Y1-c16-r1 (2026-11-21T16:22:47Z)

dsv41-5way gb300-Y1 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1886.91 | 208.52 | 0.566 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 290.74 |
| json | 195.23 |
| narrative | 114.72 |
| prose | 170.88 |
| math | 269.65 |
| reasoning | 214.64 |
| summary | 145.12 |
| format | 267.2 |
| ceiling_count | 310.49 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
