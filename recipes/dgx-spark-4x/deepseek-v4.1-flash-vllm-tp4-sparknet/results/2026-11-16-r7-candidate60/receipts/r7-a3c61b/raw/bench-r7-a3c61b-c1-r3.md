## r7-a3c61b-c1-r3 (2026-11-17T02:07:22Z)

dsv41-5way r7-a3c61b c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 105.24 | 123.71 | 0.181 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 165.55 |
| json | 154.11 |
| narrative | 72.68 |
| prose | 88.24 |
| math | 137.7 |
| reasoning | 122.32 |
| summary | 88.7 |
| format | 160.38 |
| ceiling_count | 189.05 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
