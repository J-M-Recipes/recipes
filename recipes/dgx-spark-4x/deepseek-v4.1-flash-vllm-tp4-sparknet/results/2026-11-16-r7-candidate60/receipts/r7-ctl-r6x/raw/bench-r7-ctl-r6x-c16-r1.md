## r7-ctl-r6x-c16-r1 (2026-11-17T00:50:28Z)

dsv41-5way r7-ctl-r6x c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 467.24 | 35.7 | 0.502 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 47.81 |
| json | 38.23 |
| narrative | 21.58 |
| prose | 26.41 |
| math | 41.71 |
| reasoning | 35.69 |
| summary | 26.62 |
| format | 47.56 |
| ceiling_count | 68.22 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
