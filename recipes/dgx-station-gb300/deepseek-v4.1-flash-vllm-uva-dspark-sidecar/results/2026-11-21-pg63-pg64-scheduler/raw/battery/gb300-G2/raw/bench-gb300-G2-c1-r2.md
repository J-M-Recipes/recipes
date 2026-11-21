## gb300-G2-c1-r2 (2026-11-21T14:38:36Z)

dsv41-5way gb300-G2 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 460.26 | 553.8 | 0.098 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 792.53 |
| json | 561.04 |
| narrative | 309.69 |
| prose | 334.83 |
| math | 716.17 |
| reasoning | 558.77 |
| summary | 398.12 |
| format | 759.26 |
| ceiling_count | 1039.0 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
