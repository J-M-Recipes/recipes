## x5-K1-c32-r3 (2026-11-21T16:09:56Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3707.86 | 150.21 | 0.178 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.36 |
| json | 146.25 |
| narrative | 80.61 |
| prose | 117.32 |
| math | 190.33 |
| reasoning | 150.82 |
| summary | 110.4 |
| format | 199.57 |
| ceiling_count | 218.06 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
