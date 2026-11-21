## gb300-pg62-A1-c16-r1 (2026-11-21T10:51:34Z)

dsv41-5way gb300-pg62-A1 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2392.54 | 206.75 | 0.2 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 287.17 |
| json | 188.09 |
| narrative | 116.66 |
| prose | 166.1 |
| math | 255.6 |
| reasoning | 212.01 |
| summary | 148.4 |
| format | 279.98 |
| ceiling_count | 308.92 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
