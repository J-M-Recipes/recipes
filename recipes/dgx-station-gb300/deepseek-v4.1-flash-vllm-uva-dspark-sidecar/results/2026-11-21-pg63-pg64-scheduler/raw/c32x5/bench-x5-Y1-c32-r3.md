## x5-Y1-c32-r3 (2026-11-21T16:30:56Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3654.3 | 150.35 | 0.211 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.83 |
| json | 148.21 |
| narrative | 80.81 |
| prose | 114.35 |
| math | 189.91 |
| reasoning | 154.63 |
| summary | 108.46 |
| format | 201.61 |
| ceiling_count | 218.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
