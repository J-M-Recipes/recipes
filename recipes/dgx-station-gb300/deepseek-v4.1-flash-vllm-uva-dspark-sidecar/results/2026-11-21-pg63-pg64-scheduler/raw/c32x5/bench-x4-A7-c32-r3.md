## x4-A7-c32-r3 (2026-11-21T15:03:00Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3629.49 | 149.29 | 0.189 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.61 |
| json | 144.83 |
| narrative | 80.23 |
| prose | 120.4 |
| math | 186.93 |
| reasoning | 155.01 |
| summary | 110.02 |
| format | 192.26 |
| ceiling_count | 215.15 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
