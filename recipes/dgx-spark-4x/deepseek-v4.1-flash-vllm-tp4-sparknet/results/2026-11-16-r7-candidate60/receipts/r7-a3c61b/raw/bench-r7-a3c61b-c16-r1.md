## r7-a3c61b-c16-r1 (2026-11-17T02:07:38Z)

dsv41-5way r7-a3c61b c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 528.5 | 42.63 | 0.623 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 54.07 |
| json | 45.72 |
| narrative | 23.5 |
| prose | 31.53 |
| math | 52.16 |
| reasoning | 40.41 |
| summary | 29.2 |
| format | 64.44 |
| ceiling_count | 79.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
