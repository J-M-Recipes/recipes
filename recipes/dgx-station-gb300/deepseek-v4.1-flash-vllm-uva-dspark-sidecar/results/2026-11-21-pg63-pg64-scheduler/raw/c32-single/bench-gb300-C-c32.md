## gb300-C-c32 (2026-11-21T11:28:53Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3372.73 | 144.83 | 0.271 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 204.48 |
| json | 144.13 |
| narrative | 81.15 |
| prose | 115.24 |
| math | 191.21 |
| reasoning | 151.99 |
| summary | 104.9 |
| format | 165.55 |
| ceiling_count | 217.53 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
