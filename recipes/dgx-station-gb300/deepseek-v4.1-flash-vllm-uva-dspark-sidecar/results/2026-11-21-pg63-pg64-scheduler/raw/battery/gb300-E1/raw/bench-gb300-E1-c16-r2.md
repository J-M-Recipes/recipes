## gb300-E1-c16-r2 (2026-11-21T12:56:03Z)

dsv41-5way gb300-E1 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2407.04 | 209.12 | 0.238 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.43 |
| json | 193.52 |
| narrative | 119.25 |
| prose | 162.16 |
| math | 261.55 |
| reasoning | 218.57 |
| summary | 149.98 |
| format | 274.5 |
| ceiling_count | 311.49 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
