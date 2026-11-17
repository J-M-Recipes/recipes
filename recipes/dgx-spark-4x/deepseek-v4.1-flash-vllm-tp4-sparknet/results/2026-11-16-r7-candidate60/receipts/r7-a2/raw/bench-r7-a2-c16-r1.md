## r7-a2-c16-r1 (2026-11-17T01:46:09Z)

dsv41-5way r7-a2 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 524.88 | 42.07 | 0.593 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 52.74 |
| json | 45.58 |
| narrative | 24.42 |
| prose | 31.25 |
| math | 47.97 |
| reasoning | 42.18 |
| summary | 29.26 |
| format | 63.15 |
| ceiling_count | 79.41 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
