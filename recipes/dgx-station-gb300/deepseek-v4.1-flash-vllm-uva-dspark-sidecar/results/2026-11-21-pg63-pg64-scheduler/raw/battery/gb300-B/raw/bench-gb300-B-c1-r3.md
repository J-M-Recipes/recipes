## gb300-B-c1-r3 (2026-11-21T11:07:04Z)

dsv41-5way gb300-B c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 486.48 | 559.71 | 0.036 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 794.31 |
| json | 575.56 |
| narrative | 311.72 |
| prose | 334.79 |
| math | 675.8 |
| reasoning | 557.85 |
| summary | 383.42 |
| format | 844.25 |
| ceiling_count | 1044.89 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
