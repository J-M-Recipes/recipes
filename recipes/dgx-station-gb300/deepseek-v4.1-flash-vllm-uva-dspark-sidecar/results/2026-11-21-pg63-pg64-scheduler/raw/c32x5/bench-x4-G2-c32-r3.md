## x4-G2-c32-r3 (2026-11-21T14:46:54Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3681.02 | 150.21 | 0.18 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.32 |
| json | 144.54 |
| narrative | 82.47 |
| prose | 115.99 |
| math | 196.45 |
| reasoning | 151.88 |
| summary | 107.19 |
| format | 198.84 |
| ceiling_count | 221.25 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
