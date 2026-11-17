## r7-a2-c16-r3 (2026-11-17T01:47:35Z)

dsv41-5way r7-a2 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 551.22 | 43.08 | 0.537 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 58.0 |
| json | 45.29 |
| narrative | 23.0 |
| prose | 29.89 |
| math | 53.88 |
| reasoning | 43.16 |
| summary | 28.37 |
| format | 63.02 |
| ceiling_count | 79.16 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
