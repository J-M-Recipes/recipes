## x5-K1-c32-r1 (2026-11-21T16:09:31Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3617.84 | 149.56 | 0.197 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 201.09 |
| json | 146.66 |
| narrative | 80.88 |
| prose | 115.2 |
| math | 190.07 |
| reasoning | 155.0 |
| summary | 111.65 |
| format | 195.89 |
| ceiling_count | 215.49 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
