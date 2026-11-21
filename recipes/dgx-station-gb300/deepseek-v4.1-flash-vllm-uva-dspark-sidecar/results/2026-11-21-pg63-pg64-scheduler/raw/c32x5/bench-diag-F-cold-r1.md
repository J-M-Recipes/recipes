## diag-F-cold-r1 (2026-11-21T13:53:42Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 2434.3 | 148.82 | 0.84 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 202.52 |
| json | 138.84 |
| narrative | 81.02 |
| prose | 123.38 |
| math | 190.86 |
| reasoning | 152.25 |
| summary | 105.04 |
| format | 196.65 |
| ceiling_count | 224.92 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
