## C0ctl2-c1 (2026-10-20T01:21:28Z)

Lane C ladder C0ctl2. C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.94 | 104.23 | 0.161 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 132.92 |
| json | 104.6 |
| narrative | 61.97 |
| prose | 78.74 |
| math | 126.76 |
| reasoning | 100.82 |
| summary | 78.72 |
| format | 149.32 |
| ceiling_count | 162.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
