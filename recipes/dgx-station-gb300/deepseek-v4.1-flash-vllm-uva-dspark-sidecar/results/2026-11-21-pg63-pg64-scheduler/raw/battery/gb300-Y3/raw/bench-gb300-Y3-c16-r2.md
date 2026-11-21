## gb300-Y3-c16-r2 (2026-11-21T17:41:26Z)

dsv41-5way gb300-Y3 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2401.51 | 208.87 | 0.239 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 290.6 |
| json | 195.34 |
| narrative | 118.0 |
| prose | 161.61 |
| math | 270.56 |
| reasoning | 209.27 |
| summary | 148.13 |
| format | 277.47 |
| ceiling_count | 308.25 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
