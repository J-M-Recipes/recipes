## FINAL-c1 (2026-10-19T07:43:40Z)

Lane C ladder FINAL. C1 run 1 of 2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.06 | 103.23 | 0.162 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 131.82 |
| json | 103.38 |
| narrative | 61.46 |
| prose | 78.46 |
| math | 127.03 |
| reasoning | 100.25 |
| summary | 77.78 |
| format | 145.65 |
| ceiling_count | 161.03 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
