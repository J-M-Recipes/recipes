## x5-Y3-c32-r3 (2026-11-21T17:49:24Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3566.48 | 149.33 | 0.209 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 203.11 |
| json | 144.43 |
| narrative | 82.76 |
| prose | 115.06 |
| math | 190.33 |
| reasoning | 157.4 |
| summary | 110.05 |
| format | 191.47 |
| ceiling_count | 211.75 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
