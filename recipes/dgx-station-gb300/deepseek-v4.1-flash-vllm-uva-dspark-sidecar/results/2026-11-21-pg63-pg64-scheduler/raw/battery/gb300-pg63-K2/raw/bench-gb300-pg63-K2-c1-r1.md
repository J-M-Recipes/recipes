## gb300-pg63-K2-c1-r1 (2026-11-21T17:03:29Z)

dsv41-5way gb300-pg63-K2 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 456.62 | 553.75 | 0.164 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 823.81 |
| json | 550.53 |
| narrative | 311.42 |
| prose | 334.95 |
| math | 656.85 |
| reasoning | 552.88 |
| summary | 333.49 |
| format | 866.11 |
| ceiling_count | 1018.01 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
