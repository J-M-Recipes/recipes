## diag-A5-cold-r1 (2026-11-21T14:02:38Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 2496.88 | 145.16 | 0.842 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 180.12 |
| json | 133.39 |
| narrative | 78.76 |
| prose | 115.89 |
| math | 195.12 |
| reasoning | 154.38 |
| summary | 105.08 |
| format | 198.5 |
| ceiling_count | 215.57 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
