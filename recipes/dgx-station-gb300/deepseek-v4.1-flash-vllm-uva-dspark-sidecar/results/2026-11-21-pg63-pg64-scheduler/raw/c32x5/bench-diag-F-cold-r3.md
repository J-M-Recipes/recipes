## diag-F-cold-r3 (2026-11-21T13:54:15Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3754.59 | 151.27 | 0.171 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.2 |
| json | 144.33 |
| narrative | 81.43 |
| prose | 118.85 |
| math | 196.49 |
| reasoning | 155.8 |
| summary | 113.7 |
| format | 195.38 |
| ceiling_count | 214.7 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
