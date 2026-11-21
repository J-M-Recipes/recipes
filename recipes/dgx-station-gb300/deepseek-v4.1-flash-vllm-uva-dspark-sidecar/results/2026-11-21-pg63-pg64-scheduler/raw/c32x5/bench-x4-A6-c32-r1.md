## x4-A6-c32-r1 (2026-11-21T14:13:58Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3567.35 | 149.61 | 0.221 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 203.48 |
| json | 143.47 |
| narrative | 79.2 |
| prose | 116.42 |
| math | 194.39 |
| reasoning | 150.25 |
| summary | 109.33 |
| format | 200.31 |
| ceiling_count | 216.43 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
