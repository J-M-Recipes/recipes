## r7-a3b-c1-r1 (2026-11-17T01:36:32Z)

dsv41-5way r7-a3b c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 102.92 | 120.58 | 0.182 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 164.11 |
| json | 135.91 |
| narrative | 73.9 |
| prose | 92.05 |
| math | 137.19 |
| reasoning | 105.98 |
| summary | 95.58 |
| format | 159.93 |
| ceiling_count | 187.55 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
