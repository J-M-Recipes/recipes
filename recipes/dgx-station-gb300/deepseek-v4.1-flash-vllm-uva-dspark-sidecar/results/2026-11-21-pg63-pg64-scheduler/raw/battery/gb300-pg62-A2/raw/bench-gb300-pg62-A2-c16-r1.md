## gb300-pg62-A2-c16-r1 (2026-11-21T11:53:06Z)

dsv41-5way gb300-pg62-A2 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1873.5 | 207.21 | 0.548 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 292.35 |
| json | 192.43 |
| narrative | 120.42 |
| prose | 167.63 |
| math | 262.02 |
| reasoning | 203.58 |
| summary | 140.77 |
| format | 278.49 |
| ceiling_count | 311.46 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
