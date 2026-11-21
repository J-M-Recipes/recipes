## x5-Y3-c32-r4 (2026-11-21T17:49:37Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3643.61 | 151.6 | 0.214 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 200.53 |
| json | 145.46 |
| narrative | 81.94 |
| prose | 117.01 |
| math | 197.55 |
| reasoning | 154.41 |
| summary | 111.09 |
| format | 204.78 |
| ceiling_count | 213.05 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
