## gb300-pg62-A4-c1-r3 (2026-11-21T13:25:46Z)

dsv41-5way gb300-pg62-A4 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 471.52 | 533.17 | 0.035 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 795.47 |
| json | 530.96 |
| narrative | 311.89 |
| prose | 334.61 |
| math | 633.7 |
| reasoning | 558.61 |
| summary | 350.13 |
| format | 749.97 |
| ceiling_count | 1016.95 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
