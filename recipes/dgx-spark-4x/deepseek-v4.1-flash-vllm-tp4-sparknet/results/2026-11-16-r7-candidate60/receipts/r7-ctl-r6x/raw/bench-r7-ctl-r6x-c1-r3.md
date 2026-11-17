## r7-ctl-r6x-c1-r3 (2026-11-17T00:50:10Z)

dsv41-5way r7-ctl-r6x c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 92.1 | 103.36 | 0.164 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 132.37 |
| json | 103.76 |
| narrative | 61.12 |
| prose | 77.46 |
| math | 125.73 |
| reasoning | 100.52 |
| summary | 78.06 |
| format | 147.84 |
| ceiling_count | 161.15 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
