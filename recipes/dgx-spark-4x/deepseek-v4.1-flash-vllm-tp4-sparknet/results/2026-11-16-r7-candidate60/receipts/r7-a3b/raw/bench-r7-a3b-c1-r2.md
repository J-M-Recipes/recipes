## r7-a3b-c1-r2 (2026-11-17T01:36:47Z)

dsv41-5way r7-a3b c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 104.78 | 122.72 | 0.179 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 162.94 |
| json | 150.32 |
| narrative | 72.95 |
| prose | 89.02 |
| math | 134.89 |
| reasoning | 120.22 |
| summary | 91.35 |
| format | 160.04 |
| ceiling_count | 187.4 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
