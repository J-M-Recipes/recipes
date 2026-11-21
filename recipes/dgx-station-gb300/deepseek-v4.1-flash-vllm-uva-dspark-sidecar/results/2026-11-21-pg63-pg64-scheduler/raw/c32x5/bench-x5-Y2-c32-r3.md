## x5-Y2-c32-r3 (2026-11-21T16:52:12Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3738.18 | 151.34 | 0.183 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 207.03 |
| json | 150.71 |
| narrative | 80.34 |
| prose | 114.32 |
| math | 193.76 |
| reasoning | 155.23 |
| summary | 111.55 |
| format | 197.77 |
| ceiling_count | 216.83 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
