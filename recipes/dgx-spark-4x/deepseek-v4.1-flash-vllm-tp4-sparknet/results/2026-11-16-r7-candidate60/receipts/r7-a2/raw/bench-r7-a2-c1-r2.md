## r7-a2-c1-r2 (2026-11-17T01:45:38Z)

dsv41-5way r7-a2 c1 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 105.36 | 123.18 | 0.174 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 164.11 |
| json | 151.36 |
| narrative | 72.0 |
| prose | 89.5 |
| math | 135.73 |
| reasoning | 119.93 |
| summary | 91.57 |
| format | 161.23 |
| ceiling_count | 188.58 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
