## gb300-E2-c16-r1 (2026-11-21T13:10:49Z)

dsv41-5way gb300-E2 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1921.52 | 210.34 | 0.557 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 291.0 |
| json | 201.45 |
| narrative | 118.8 |
| prose | 168.07 |
| math | 266.83 |
| reasoning | 217.93 |
| summary | 144.28 |
| format | 274.37 |
| ceiling_count | 309.69 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
