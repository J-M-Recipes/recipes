## x4-G2-c32-r4 (2026-11-21T14:47:07Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3667.51 | 149.04 | 0.181 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.74 |
| json | 141.75 |
| narrative | 82.15 |
| prose | 117.69 |
| math | 192.36 |
| reasoning | 152.72 |
| summary | 105.51 |
| format | 195.4 |
| ceiling_count | 217.52 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
