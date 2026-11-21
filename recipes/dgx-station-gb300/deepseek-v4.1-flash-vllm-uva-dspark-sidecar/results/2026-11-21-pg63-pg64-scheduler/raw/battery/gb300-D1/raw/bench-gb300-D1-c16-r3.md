## gb300-D1-c16-r3 (2026-11-21T12:10:59Z)

dsv41-5way gb300-D1 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2420.7 | 207.48 | 0.178 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 291.28 |
| json | 192.29 |
| narrative | 118.95 |
| prose | 164.98 |
| math | 244.76 |
| reasoning | 216.85 |
| summary | 151.54 |
| format | 279.16 |
| ceiling_count | 310.39 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
