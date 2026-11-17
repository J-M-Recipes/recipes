## r7-a3-c1-r1 (2026-11-17T02:06:14Z)

dsv41-5way r7-a3 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 105.03 | 123.18 | 0.182 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 164.42 |
| json | 137.18 |
| narrative | 73.75 |
| prose | 92.4 |
| math | 137.66 |
| reasoning | 123.71 |
| summary | 95.9 |
| format | 160.45 |
| ceiling_count | 187.9 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
