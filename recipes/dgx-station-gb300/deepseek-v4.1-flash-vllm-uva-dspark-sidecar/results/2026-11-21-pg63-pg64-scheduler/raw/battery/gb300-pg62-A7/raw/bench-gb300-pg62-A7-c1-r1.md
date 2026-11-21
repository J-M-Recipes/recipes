## gb300-pg62-A7-c1-r1 (2026-11-21T14:54:37Z)

dsv41-5way gb300-pg62-A7 c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 444.69 | 536.9 | 0.171 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.84 |
| json | 654.68 |
| narrative | 310.76 |
| prose | 333.68 |
| math | 611.04 |
| reasoning | 553.95 |
| summary | 290.34 |
| format | 713.95 |
| ceiling_count | 1023.78 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
