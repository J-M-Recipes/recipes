## gb300-C-c1-r1 (2026-11-21T11:21:55Z)

dsv41-5way gb300-C c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 456.53 | 588.6 | 0.17 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 1080.62 |
| json | 563.21 |
| narrative | 311.44 |
| prose | 334.73 |
| math | 625.28 |
| reasoning | 567.86 |
| summary | 339.21 |
| format | 886.45 |
| ceiling_count | 1020.9 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
