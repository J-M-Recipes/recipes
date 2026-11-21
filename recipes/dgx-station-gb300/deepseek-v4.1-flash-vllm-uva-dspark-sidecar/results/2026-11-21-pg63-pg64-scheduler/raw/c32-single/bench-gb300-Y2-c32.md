## gb300-Y2-c32 (2026-11-21T16:50:47Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3504.34 | 149.63 | 0.268 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 208.64 |
| json | 144.3 |
| narrative | 79.69 |
| prose | 116.12 |
| math | 187.06 |
| reasoning | 156.84 |
| summary | 105.0 |
| format | 199.37 |
| ceiling_count | 217.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
