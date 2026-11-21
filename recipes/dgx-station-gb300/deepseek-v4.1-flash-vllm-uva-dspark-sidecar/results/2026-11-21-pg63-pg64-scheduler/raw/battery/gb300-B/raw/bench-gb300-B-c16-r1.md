## gb300-B-c16-r1 (2026-11-21T11:07:08Z)

dsv41-5way gb300-B c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1875.73 | 202.22 | 0.533 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 281.01 |
| json | 189.22 |
| narrative | 119.9 |
| prose | 162.22 |
| math | 258.93 |
| reasoning | 211.91 |
| summary | 139.67 |
| format | 254.92 |
| ceiling_count | 310.72 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
