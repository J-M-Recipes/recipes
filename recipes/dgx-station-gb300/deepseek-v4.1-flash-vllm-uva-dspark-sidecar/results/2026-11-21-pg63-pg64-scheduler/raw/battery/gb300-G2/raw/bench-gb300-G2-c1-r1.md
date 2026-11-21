## gb300-G2-c1-r1 (2026-11-21T14:38:31Z)

dsv41-5way gb300-G2 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 452.84 | 550.21 | 0.165 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 775.51 |
| json | 535.0 |
| narrative | 310.83 |
| prose | 334.46 |
| math | 665.38 |
| reasoning | 552.75 |
| summary | 355.43 |
| format | 872.31 |
| ceiling_count | 1017.11 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
