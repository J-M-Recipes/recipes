## x4-G1-c32-r5 (2026-11-21T14:31:14Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3645.14 | 150.03 | 0.2 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 208.57 |
| json | 146.9 |
| narrative | 81.7 |
| prose | 115.33 |
| math | 191.02 |
| reasoning | 150.55 |
| summary | 108.73 |
| format | 197.41 |
| ceiling_count | 218.01 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
