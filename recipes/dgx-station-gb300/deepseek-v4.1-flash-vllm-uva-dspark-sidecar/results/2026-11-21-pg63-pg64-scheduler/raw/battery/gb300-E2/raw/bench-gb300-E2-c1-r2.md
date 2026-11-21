## gb300-E2-c1-r2 (2026-11-21T13:10:41Z)

dsv41-5way gb300-E2 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 459.11 | 551.72 | 0.097 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.31 |
| json | 583.56 |
| narrative | 311.81 |
| prose | 334.38 |
| math | 653.42 |
| reasoning | 559.1 |
| summary | 390.2 |
| format | 787.98 |
| ceiling_count | 1044.0 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
