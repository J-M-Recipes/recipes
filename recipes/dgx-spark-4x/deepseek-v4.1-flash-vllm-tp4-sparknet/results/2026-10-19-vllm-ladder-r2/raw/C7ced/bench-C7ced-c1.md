## C7ced-c1 (2026-10-20T01:36:41Z)

Lane C ladder C7ced. C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.88 | 104.09 | 0.161 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 133.6 |
| json | 104.07 |
| narrative | 62.22 |
| prose | 78.89 |
| math | 126.88 |
| reasoning | 100.79 |
| summary | 78.42 |
| format | 147.89 |
| ceiling_count | 161.53 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
