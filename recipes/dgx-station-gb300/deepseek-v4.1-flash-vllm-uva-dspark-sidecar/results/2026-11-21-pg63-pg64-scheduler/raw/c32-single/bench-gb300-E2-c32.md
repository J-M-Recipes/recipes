## gb300-E2-c32 (2026-11-21T13:17:31Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3532.73 | 150.64 | 0.283 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.12 |
| json | 143.94 |
| narrative | 81.84 |
| prose | 116.77 |
| math | 192.69 |
| reasoning | 158.19 |
| summary | 105.42 |
| format | 200.14 |
| ceiling_count | 217.75 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
