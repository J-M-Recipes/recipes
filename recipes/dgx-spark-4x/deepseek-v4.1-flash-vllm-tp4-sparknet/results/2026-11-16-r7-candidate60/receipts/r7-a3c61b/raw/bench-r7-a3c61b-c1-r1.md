## r7-a3c61b-c1-r1 (2026-11-17T02:06:51Z)

dsv41-5way r7-a3c61b c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 105.97 | 125.33 | 0.186 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 163.23 |
| json | 156.64 |
| narrative | 74.4 |
| prose | 91.57 |
| math | 138.04 |
| reasoning | 124.06 |
| summary | 93.91 |
| format | 160.8 |
| ceiling_count | 188.54 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
