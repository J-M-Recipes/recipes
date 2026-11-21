## x4-A7-c32-r1 (2026-11-21T15:02:34Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3567.14 | 148.38 | 0.207 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.02 |
| json | 141.67 |
| narrative | 83.65 |
| prose | 113.91 |
| math | 189.47 |
| reasoning | 150.75 |
| summary | 107.77 |
| format | 195.81 |
| ceiling_count | 222.25 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
