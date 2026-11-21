## gb300-D1-c1-r3 (2026-11-21T12:10:33Z)

dsv41-5way gb300-D1 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 478.21 | 551.53 | 0.036 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.87 |
| json | 579.4 |
| narrative | 311.91 |
| prose | 335.22 |
| math | 605.44 |
| reasoning | 558.44 |
| summary | 380.31 |
| format | 847.66 |
| ceiling_count | 1015.74 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
