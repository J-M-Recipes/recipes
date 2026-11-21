## x4-G1-c32-r1 (2026-11-21T14:30:23Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3629.45 | 149.52 | 0.211 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.8 |
| json | 144.58 |
| narrative | 82.8 |
| prose | 119.42 |
| math | 185.89 |
| reasoning | 154.06 |
| summary | 109.46 |
| format | 194.11 |
| ceiling_count | 216.95 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
