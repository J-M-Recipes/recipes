## x4-A6-c32-r5 (2026-11-21T14:14:48Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3655.0 | 149.3 | 0.194 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 207.55 |
| json | 139.99 |
| narrative | 81.09 |
| prose | 116.43 |
| math | 190.56 |
| reasoning | 151.54 |
| summary | 109.84 |
| format | 197.43 |
| ceiling_count | 215.7 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
