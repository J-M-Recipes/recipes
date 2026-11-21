## gb300-pg62-A6-c1-r1 (2026-11-21T14:06:06Z)

dsv41-5way gb300-pg62-A6 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 469.63 | 541.25 | 0.046 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.0 |
| json | 475.36 |
| narrative | 311.63 |
| prose | 341.71 |
| math | 623.37 |
| reasoning | 570.12 |
| summary | 355.14 |
| format | 826.71 |
| ceiling_count | 1021.22 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
