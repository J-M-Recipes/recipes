## x5-Y1-c32-r1 (2026-11-21T16:30:30Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3615.68 | 149.57 | 0.208 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 209.03 |
| json | 141.75 |
| narrative | 81.76 |
| prose | 116.91 |
| math | 197.88 |
| reasoning | 149.85 |
| summary | 109.49 |
| format | 189.9 |
| ceiling_count | 222.03 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
