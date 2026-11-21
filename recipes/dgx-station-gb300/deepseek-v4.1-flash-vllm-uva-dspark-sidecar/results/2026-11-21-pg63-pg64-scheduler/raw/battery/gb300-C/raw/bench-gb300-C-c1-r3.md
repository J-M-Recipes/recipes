## gb300-C-c1-r3 (2026-11-21T11:22:05Z)

dsv41-5way gb300-C c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 491.51 | 562.48 | 0.033 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.78 |
| json | 544.39 |
| narrative | 311.88 |
| prose | 334.65 |
| math | 665.31 |
| reasoning | 558.33 |
| summary | 392.74 |
| format | 897.76 |
| ceiling_count | 1043.0 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
