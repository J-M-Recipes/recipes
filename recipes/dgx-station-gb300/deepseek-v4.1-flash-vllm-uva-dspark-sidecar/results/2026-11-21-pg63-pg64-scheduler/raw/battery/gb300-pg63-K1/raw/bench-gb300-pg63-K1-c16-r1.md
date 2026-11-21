## gb300-pg63-K1-c16-r1 (2026-11-21T16:01:47Z)

dsv41-5way gb300-pg63-K1 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2129.95 | 203.72 | 0.386 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 282.8 |
| json | 191.57 |
| narrative | 120.45 |
| prose | 159.88 |
| math | 269.69 |
| reasoning | 211.29 |
| summary | 151.11 |
| format | 242.98 |
| ceiling_count | 308.22 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
