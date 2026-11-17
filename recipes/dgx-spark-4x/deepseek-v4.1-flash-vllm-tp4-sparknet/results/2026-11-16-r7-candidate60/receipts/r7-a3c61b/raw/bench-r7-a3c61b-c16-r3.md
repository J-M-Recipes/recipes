## r7-a3c61b-c16-r3 (2026-11-17T02:09:06Z)

dsv41-5way r7-a3c61b c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 555.23 | 43.05 | 0.511 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 58.57 |
| json | 44.71 |
| narrative | 22.78 |
| prose | 30.4 |
| math | 52.84 |
| reasoning | 41.8 |
| summary | 28.38 |
| format | 64.92 |
| ceiling_count | 79.05 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
