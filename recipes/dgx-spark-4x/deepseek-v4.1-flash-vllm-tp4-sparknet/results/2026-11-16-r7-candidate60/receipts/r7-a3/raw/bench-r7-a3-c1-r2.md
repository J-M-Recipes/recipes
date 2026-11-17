## r7-a3-c1-r2 (2026-11-17T02:06:29Z)

dsv41-5way r7-a3 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 105.12 | 123.02 | 0.176 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 163.53 |
| json | 151.1 |
| narrative | 73.0 |
| prose | 89.33 |
| math | 135.33 |
| reasoning | 119.99 |
| summary | 91.64 |
| format | 160.25 |
| ceiling_count | 187.96 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
