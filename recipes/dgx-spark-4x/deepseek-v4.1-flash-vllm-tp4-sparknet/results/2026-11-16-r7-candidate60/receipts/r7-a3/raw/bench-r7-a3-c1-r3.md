## r7-a3-c1-r3 (2026-11-17T02:06:44Z)

dsv41-5way r7-a3 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 106.58 | 125.36 | 0.179 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 165.34 |
| json | 155.59 |
| narrative | 73.52 |
| prose | 90.74 |
| math | 137.57 |
| reasoning | 123.72 |
| summary | 96.12 |
| format | 160.3 |
| ceiling_count | 187.81 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
