## gb300-Y3-c16-r1 (2026-11-21T17:41:14Z)

dsv41-5way gb300-Y3 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1893.1 | 210.8 | 0.569 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 289.51 |
| json | 197.78 |
| narrative | 119.77 |
| prose | 164.7 |
| math | 278.61 |
| reasoning | 218.39 |
| summary | 149.53 |
| format | 268.13 |
| ceiling_count | 311.41 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
