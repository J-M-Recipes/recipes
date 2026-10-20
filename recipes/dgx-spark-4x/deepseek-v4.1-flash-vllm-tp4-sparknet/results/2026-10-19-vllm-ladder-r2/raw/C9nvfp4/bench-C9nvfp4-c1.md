## C9nvfp4-c1 (2026-10-20T01:58:41Z)

Lane C ladder C9nvfp4. C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 94.3 | 105.96 | 0.163 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 138.67 |
| json | 102.99 |
| narrative | 65.36 |
| prose | 79.35 |
| math | 129.58 |
| reasoning | 100.86 |
| summary | 78.84 |
| format | 152.05 |
| ceiling_count | 164.91 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
