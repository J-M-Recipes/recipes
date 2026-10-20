## C8pp2-c1 (2026-10-20T01:47:17Z)

Lane C ladder C8pp2. C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 93.0 | 104.4 | 0.162 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 133.9 |
| json | 103.83 |
| narrative | 62.22 |
| prose | 78.37 |
| math | 127.77 |
| reasoning | 101.48 |
| summary | 79.02 |
| format | 148.63 |
| ceiling_count | 162.68 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
