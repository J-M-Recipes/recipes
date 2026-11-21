## x4-A6-c32-r3 (2026-11-21T14:14:23Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3717.89 | 151.14 | 0.191 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.14 |
| json | 146.37 |
| narrative | 80.82 |
| prose | 120.24 |
| math | 192.72 |
| reasoning | 154.69 |
| summary | 109.04 |
| format | 200.09 |
| ceiling_count | 215.12 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
