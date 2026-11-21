## gb300-C-c16-r2 (2026-11-21T11:22:21Z)

dsv41-5way gb300-C c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2406.16 | 206.68 | 0.228 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 284.64 |
| json | 192.53 |
| narrative | 117.94 |
| prose | 162.77 |
| math | 274.76 |
| reasoning | 207.53 |
| summary | 145.02 |
| format | 268.21 |
| ceiling_count | 301.87 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
