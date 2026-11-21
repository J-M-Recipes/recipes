## gb300-pg62-A7-c16-r3 (2026-11-21T14:55:13Z)

dsv41-5way gb300-pg62-A7 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2434.07 | 208.57 | 0.18 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 289.75 |
| json | 197.93 |
| narrative | 115.92 |
| prose | 167.71 |
| math | 259.1 |
| reasoning | 209.02 |
| summary | 150.41 |
| format | 278.73 |
| ceiling_count | 312.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
