## r7-ctl-r6x-c16-r2 (2026-11-17T00:51:16Z)

dsv41-5way r7-ctl-r6x c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 479.24 | 36.4 | 0.489 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 49.07 |
| json | 38.24 |
| narrative | 21.6 |
| prose | 26.38 |
| math | 44.87 |
| reasoning | 35.64 |
| summary | 24.8 |
| format | 50.58 |
| ceiling_count | 68.08 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
