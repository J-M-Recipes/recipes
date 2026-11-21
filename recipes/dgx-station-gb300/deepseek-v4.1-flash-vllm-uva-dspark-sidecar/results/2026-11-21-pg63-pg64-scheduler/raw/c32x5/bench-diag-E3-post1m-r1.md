## diag-E3-post1m-r1 (2026-11-21T13:45:54Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3761.25 | 152.41 | 0.181 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.17 |
| json | 147.94 |
| narrative | 82.36 |
| prose | 118.8 |
| math | 193.54 |
| reasoning | 153.72 |
| summary | 109.16 |
| format | 208.6 |
| ceiling_count | 216.36 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
