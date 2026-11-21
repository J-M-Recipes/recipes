## gb300-pg63-K2-c16-r3 (2026-11-21T17:04:05Z)

dsv41-5way gb300-pg63-K2 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2468.16 | 214.74 | 0.185 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 295.84 |
| json | 197.43 |
| narrative | 120.34 |
| prose | 169.33 |
| math | 274.7 |
| reasoning | 216.0 |
| summary | 152.24 |
| format | 292.0 |
| ceiling_count | 312.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
