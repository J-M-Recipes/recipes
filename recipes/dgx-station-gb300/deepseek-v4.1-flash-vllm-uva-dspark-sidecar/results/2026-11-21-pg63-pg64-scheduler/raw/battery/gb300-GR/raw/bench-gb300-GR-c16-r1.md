## gb300-GR-c16-r1 (2026-11-21T11:37:29Z)

dsv41-5way gb300-GR c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1917.44 | 210.29 | 0.547 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.65 |
| json | 194.85 |
| narrative | 122.74 |
| prose | 166.0 |
| math | 275.96 |
| reasoning | 211.09 |
| summary | 145.59 |
| format | 272.45 |
| ceiling_count | 311.61 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
