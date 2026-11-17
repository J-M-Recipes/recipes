## r7-a2-c1-r1 (2026-11-17T01:45:22Z)

dsv41-5way r7-a2 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 103.54 | 121.22 | 0.18 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 164.37 |
| json | 137.58 |
| narrative | 74.25 |
| prose | 92.61 |
| math | 138.42 |
| reasoning | 106.48 |
| summary | 95.31 |
| format | 160.74 |
| ceiling_count | 188.9 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
