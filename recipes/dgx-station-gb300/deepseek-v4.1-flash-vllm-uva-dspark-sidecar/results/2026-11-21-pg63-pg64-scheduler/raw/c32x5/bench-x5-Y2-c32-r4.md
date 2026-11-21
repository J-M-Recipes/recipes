## x5-Y2-c32-r4 (2026-11-21T16:52:25Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3637.48 | 149.66 | 0.194 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 205.5 |
| json | 146.82 |
| narrative | 81.54 |
| prose | 113.91 |
| math | 195.49 |
| reasoning | 155.12 |
| summary | 106.79 |
| format | 192.09 |
| ceiling_count | 218.24 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
