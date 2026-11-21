## x5-Y3-c32-r5 (2026-11-21T17:49:50Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3635.07 | 149.61 | 0.191 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.93 |
| json | 148.93 |
| narrative | 83.26 |
| prose | 116.2 |
| math | 188.49 |
| reasoning | 151.42 |
| summary | 108.85 |
| format | 193.82 |
| ceiling_count | 216.68 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
