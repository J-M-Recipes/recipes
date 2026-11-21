## gb300-pg62-A4-c1-r2 (2026-11-21T13:25:42Z)

dsv41-5way gb300-pg62-A4 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 452.91 | 540.37 | 0.1 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 793.58 |
| json | 567.23 |
| narrative | 311.68 |
| prose | 334.96 |
| math | 638.01 |
| reasoning | 559.21 |
| summary | 361.23 |
| format | 757.07 |
| ceiling_count | 1018.24 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
