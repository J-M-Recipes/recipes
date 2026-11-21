## gb300-Y2-c16-r1 (2026-11-21T16:44:03Z)

dsv41-5way gb300-Y2 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1899.04 | 208.76 | 0.555 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.95 |
| json | 197.0 |
| narrative | 121.5 |
| prose | 162.92 |
| math | 262.94 |
| reasoning | 208.75 |
| summary | 145.37 |
| format | 277.67 |
| ceiling_count | 311.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
