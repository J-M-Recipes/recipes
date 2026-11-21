## x4-A7-c32-r4 (2026-11-21T15:03:12Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3693.99 | 151.3 | 0.194 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.98 |
| json | 145.6 |
| narrative | 81.39 |
| prose | 118.24 |
| math | 196.02 |
| reasoning | 151.0 |
| summary | 114.77 |
| format | 197.4 |
| ceiling_count | 216.78 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
