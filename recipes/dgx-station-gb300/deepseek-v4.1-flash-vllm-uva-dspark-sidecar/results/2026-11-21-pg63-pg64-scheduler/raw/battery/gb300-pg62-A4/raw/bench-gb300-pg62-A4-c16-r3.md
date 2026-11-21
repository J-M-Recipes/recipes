## gb300-pg62-A4-c16-r3 (2026-11-21T13:26:13Z)

dsv41-5way gb300-pg62-A4 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2429.38 | 211.58 | 0.237 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 296.8 |
| json | 195.32 |
| narrative | 118.14 |
| prose | 168.0 |
| math | 279.35 |
| reasoning | 213.59 |
| summary | 149.94 |
| format | 271.47 |
| ceiling_count | 311.66 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
