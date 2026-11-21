## gb300-pg62-A7-c32 (2026-11-21T15:01:35Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3600.14 | 149.1 | 0.248 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 206.12 |
| json | 144.73 |
| narrative | 81.59 |
| prose | 117.4 |
| math | 186.47 |
| reasoning | 153.29 |
| summary | 101.58 |
| format | 201.66 |
| ceiling_count | 218.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
