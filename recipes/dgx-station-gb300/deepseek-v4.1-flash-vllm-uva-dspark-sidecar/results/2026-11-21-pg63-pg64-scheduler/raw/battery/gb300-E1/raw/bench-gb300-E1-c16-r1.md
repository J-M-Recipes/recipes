## gb300-E1-c16-r1 (2026-11-21T12:55:50Z)

dsv41-5way gb300-E1 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1904.54 | 212.63 | 0.569 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 297.68 |
| json | 193.72 |
| narrative | 123.59 |
| prose | 163.93 |
| math | 280.34 |
| reasoning | 212.82 |
| summary | 151.37 |
| format | 277.56 |
| ceiling_count | 310.33 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
