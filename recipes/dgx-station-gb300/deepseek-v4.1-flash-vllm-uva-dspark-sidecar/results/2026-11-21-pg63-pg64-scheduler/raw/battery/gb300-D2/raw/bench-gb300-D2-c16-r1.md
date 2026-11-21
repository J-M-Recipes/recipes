## gb300-D2-c16-r1 (2026-11-21T12:25:41Z)

dsv41-5way gb300-D2 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1894.59 | 211.86 | 0.568 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 291.79 |
| json | 195.86 |
| narrative | 119.6 |
| prose | 162.65 |
| math | 290.88 |
| reasoning | 205.48 |
| summary | 148.76 |
| format | 279.85 |
| ceiling_count | 307.92 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
