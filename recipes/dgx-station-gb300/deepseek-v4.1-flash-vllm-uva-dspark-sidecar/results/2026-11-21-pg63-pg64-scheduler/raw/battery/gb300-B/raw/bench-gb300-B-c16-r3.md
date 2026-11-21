## gb300-B-c16-r3 (2026-11-21T11:07:30Z)

dsv41-5way gb300-B c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2459.21 | 208.89 | 0.173 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 300.53 |
| json | 191.56 |
| narrative | 119.39 |
| prose | 169.0 |
| math | 266.67 |
| reasoning | 204.9 |
| summary | 147.52 |
| format | 271.57 |
| ceiling_count | 311.55 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
