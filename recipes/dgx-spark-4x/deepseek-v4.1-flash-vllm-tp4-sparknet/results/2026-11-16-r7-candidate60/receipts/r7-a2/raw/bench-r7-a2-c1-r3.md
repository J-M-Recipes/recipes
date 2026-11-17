## r7-a2-c1-r3 (2026-11-17T01:45:53Z)

dsv41-5way r7-a2 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 106.85 | 125.68 | 0.178 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 166.24 |
| json | 155.9 |
| narrative | 73.95 |
| prose | 91.0 |
| math | 138.12 |
| reasoning | 123.42 |
| summary | 96.26 |
| format | 160.56 |
| ceiling_count | 187.6 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
