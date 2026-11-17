## r7-ctl-r6x-c1-r2 (2026-11-17T00:49:52Z)

dsv41-5way r7-ctl-r6x c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.06 | 103.33 | 0.164 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 132.39 |
| json | 103.85 |
| narrative | 61.16 |
| prose | 77.44 |
| math | 125.95 |
| reasoning | 100.14 |
| summary | 78.15 |
| format | 147.59 |
| ceiling_count | 160.63 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
