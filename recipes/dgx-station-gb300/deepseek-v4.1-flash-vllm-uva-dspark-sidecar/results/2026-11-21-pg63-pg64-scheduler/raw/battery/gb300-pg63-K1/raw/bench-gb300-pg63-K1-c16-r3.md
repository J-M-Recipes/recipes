## gb300-pg63-K1-c16-r3 (2026-11-21T16:02:09Z)

dsv41-5way gb300-pg63-K1 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2436.07 | 210.38 | 0.184 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 290.22 |
| json | 197.42 |
| narrative | 120.39 |
| prose | 162.89 |
| math | 270.79 |
| reasoning | 210.64 |
| summary | 154.18 |
| format | 276.54 |
| ceiling_count | 309.6 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
