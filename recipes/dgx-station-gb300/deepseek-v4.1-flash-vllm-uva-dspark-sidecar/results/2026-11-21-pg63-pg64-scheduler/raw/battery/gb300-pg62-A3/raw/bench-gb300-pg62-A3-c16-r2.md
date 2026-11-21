## gb300-pg62-A3-c16-r2 (2026-11-21T12:40:54Z)

dsv41-5way gb300-pg62-A3 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2386.35 | 208.94 | 0.237 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 295.75 |
| json | 193.45 |
| narrative | 117.92 |
| prose | 161.65 |
| math | 274.03 |
| reasoning | 204.61 |
| summary | 151.99 |
| format | 272.15 |
| ceiling_count | 310.41 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
