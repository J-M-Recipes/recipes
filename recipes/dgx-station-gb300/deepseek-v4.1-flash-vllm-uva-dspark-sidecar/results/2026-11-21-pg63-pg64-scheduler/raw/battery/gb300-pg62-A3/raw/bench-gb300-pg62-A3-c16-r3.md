## gb300-pg62-A3-c16-r3 (2026-11-21T12:41:04Z)

dsv41-5way gb300-pg62-A3 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2453.18 | 208.54 | 0.174 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.09 |
| json | 190.98 |
| narrative | 120.44 |
| prose | 160.97 |
| math | 266.74 |
| reasoning | 210.89 |
| summary | 154.23 |
| format | 271.0 |
| ceiling_count | 310.61 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
