## diag-E3-cold-r1 (2026-11-21T13:44:26Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 2594.14 | 144.84 | 0.704 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 171.12 |
| json | 145.48 |
| narrative | 80.45 |
| prose | 115.76 |
| math | 193.42 |
| reasoning | 148.81 |
| summary | 106.91 |
| format | 196.73 |
| ceiling_count | 213.52 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
