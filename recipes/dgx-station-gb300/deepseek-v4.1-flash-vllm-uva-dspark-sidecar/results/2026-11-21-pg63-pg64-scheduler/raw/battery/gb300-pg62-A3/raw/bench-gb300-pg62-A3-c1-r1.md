## gb300-pg62-A3-c1-r1 (2026-11-21T12:40:28Z)

dsv41-5way gb300-pg62-A3 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 460.07 | 563.18 | 0.166 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.49 |
| json | 583.54 |
| narrative | 312.39 |
| prose | 334.68 |
| math | 601.75 |
| reasoning | 556.74 |
| summary | 337.81 |
| format | 952.04 |
| ceiling_count | 1020.25 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
