## gb300-B-c32 (2026-11-21T11:13:52Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3470.96 | 148.78 | 0.273 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 201.41 |
| json | 141.48 |
| narrative | 79.78 |
| prose | 118.69 |
| math | 189.82 |
| reasoning | 156.62 |
| summary | 107.06 |
| format | 195.34 |
| ceiling_count | 224.3 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
