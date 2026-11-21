## diag-E3-cold-r3 (2026-11-21T13:44:58Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3688.41 | 150.96 | 0.195 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.9 |
| json | 139.15 |
| narrative | 81.61 |
| prose | 119.24 |
| math | 196.98 |
| reasoning | 154.24 |
| summary | 108.93 |
| format | 201.59 |
| ceiling_count | 226.41 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
