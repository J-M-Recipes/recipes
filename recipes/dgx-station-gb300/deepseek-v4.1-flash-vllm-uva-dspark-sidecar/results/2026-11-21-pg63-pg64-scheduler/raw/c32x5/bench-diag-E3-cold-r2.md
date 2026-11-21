## diag-E3-cold-r2 (2026-11-21T13:44:45Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3649.82 | 150.74 | 0.247 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 207.0 |
| json | 144.67 |
| narrative | 81.27 |
| prose | 116.88 |
| math | 192.59 |
| reasoning | 152.3 |
| summary | 109.84 |
| format | 201.37 |
| ceiling_count | 215.92 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
