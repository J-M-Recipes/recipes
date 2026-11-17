## r7-a3b-c1-r3 (2026-11-17T01:37:03Z)

dsv41-5way r7-a3b c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 106.24 | 124.94 | 0.18 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 164.53 |
| json | 155.21 |
| narrative | 73.46 |
| prose | 90.61 |
| math | 137.13 |
| reasoning | 123.38 |
| summary | 95.84 |
| format | 159.39 |
| ceiling_count | 187.53 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
