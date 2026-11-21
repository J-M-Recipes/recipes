## gb300-G1-c32 (2026-11-21T14:29:23Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3511.04 | 149.68 | 0.27 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.55 |
| json | 143.66 |
| narrative | 81.34 |
| prose | 117.0 |
| math | 186.45 |
| reasoning | 152.3 |
| summary | 106.73 |
| format | 203.37 |
| ceiling_count | 215.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
