## gb300-G2-c16-r1 (2026-11-21T14:38:44Z)

dsv41-5way gb300-G2 c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1891.12 | 209.16 | 0.557 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 298.89 |
| json | 193.8 |
| narrative | 121.72 |
| prose | 165.91 |
| math | 272.68 |
| reasoning | 209.29 |
| summary | 143.33 |
| format | 267.63 |
| ceiling_count | 312.03 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
