## r7-ctl-r6x-c16-r3 (2026-11-17T00:52:04Z)

dsv41-5way r7-ctl-r6x c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 471.76 | 35.57 | 0.485 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 47.02 |
| json | 38.26 |
| narrative | 21.51 |
| prose | 26.49 |
| math | 44.8 |
| reasoning | 35.46 |
| summary | 23.95 |
| format | 47.06 |
| ceiling_count | 64.37 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
