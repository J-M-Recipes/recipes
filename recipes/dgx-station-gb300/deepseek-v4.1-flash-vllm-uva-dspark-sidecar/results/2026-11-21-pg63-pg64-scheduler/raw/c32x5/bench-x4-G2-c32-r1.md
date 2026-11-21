## x4-G2-c32-r1 (2026-11-21T14:46:29Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3528.36 | 148.74 | 0.22 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.84 |
| json | 141.2 |
| narrative | 82.08 |
| prose | 119.63 |
| math | 185.58 |
| reasoning | 149.92 |
| summary | 108.36 |
| format | 198.27 |
| ceiling_count | 217.24 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
