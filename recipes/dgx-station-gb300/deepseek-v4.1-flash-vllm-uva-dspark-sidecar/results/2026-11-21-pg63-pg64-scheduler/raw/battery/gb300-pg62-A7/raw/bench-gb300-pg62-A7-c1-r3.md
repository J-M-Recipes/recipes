## gb300-pg62-A7-c1-r3 (2026-11-21T14:54:47Z)

dsv41-5way gb300-pg62-A7 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 487.28 | 562.92 | 0.036 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 796.15 |
| json | 528.23 |
| narrative | 311.74 |
| prose | 334.02 |
| math | 700.66 |
| reasoning | 566.35 |
| summary | 395.53 |
| format | 870.7 |
| ceiling_count | 1072.36 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
