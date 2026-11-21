## diag-F-cold-r2 (2026-11-21T13:54:01Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3486.28 | 148.96 | 0.313 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 202.78 |
| json | 146.29 |
| narrative | 79.95 |
| prose | 116.39 |
| math | 188.93 |
| reasoning | 152.31 |
| summary | 105.88 |
| format | 199.18 |
| ceiling_count | 213.68 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
