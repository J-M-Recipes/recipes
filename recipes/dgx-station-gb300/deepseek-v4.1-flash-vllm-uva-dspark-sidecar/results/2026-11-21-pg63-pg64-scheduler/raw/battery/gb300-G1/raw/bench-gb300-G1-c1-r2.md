## gb300-G1-c1-r2 (2026-11-21T14:22:31Z)

dsv41-5way gb300-G1 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 455.68 | 543.76 | 0.101 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.21 |
| json | 655.65 |
| narrative | 311.71 |
| prose | 334.51 |
| math | 627.46 |
| reasoning | 559.21 |
| summary | 319.0 |
| format | 748.32 |
| ceiling_count | 1016.28 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
