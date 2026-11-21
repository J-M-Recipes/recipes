## diag-E3-post1m-r2 (2026-11-21T13:46:06Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3729.14 | 151.08 | 0.176 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.78 |
| json | 145.78 |
| narrative | 82.53 |
| prose | 115.67 |
| math | 194.38 |
| reasoning | 154.43 |
| summary | 110.35 |
| format | 199.71 |
| ceiling_count | 215.98 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
