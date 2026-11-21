## gb300-G2-c32 (2026-11-21T14:45:29Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3610.06 | 151.05 | 0.284 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 208.47 |
| json | 144.74 |
| narrative | 82.46 |
| prose | 117.95 |
| math | 196.16 |
| reasoning | 154.16 |
| summary | 105.13 |
| format | 199.32 |
| ceiling_count | 224.51 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
