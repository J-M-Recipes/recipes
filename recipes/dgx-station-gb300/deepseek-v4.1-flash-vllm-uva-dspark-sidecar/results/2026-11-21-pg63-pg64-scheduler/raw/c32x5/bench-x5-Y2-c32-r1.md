## x5-Y2-c32-r1 (2026-11-21T16:51:47Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3581.35 | 149.84 | 0.207 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 202.34 |
| json | 146.34 |
| narrative | 81.2 |
| prose | 116.05 |
| math | 190.17 |
| reasoning | 153.41 |
| summary | 110.3 |
| format | 198.92 |
| ceiling_count | 225.91 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
