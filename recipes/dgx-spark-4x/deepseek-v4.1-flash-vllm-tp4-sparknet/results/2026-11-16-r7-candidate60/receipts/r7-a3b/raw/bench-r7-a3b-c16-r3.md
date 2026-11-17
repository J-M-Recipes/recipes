## r7-a3b-c16-r3 (2026-11-17T01:38:45Z)

dsv41-5way r7-a3b c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 541.56 | 42.44 | 0.52 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 55.21 |
| json | 46.54 |
| narrative | 22.9 |
| prose | 30.22 |
| math | 52.67 |
| reasoning | 41.27 |
| summary | 28.24 |
| format | 62.49 |
| ceiling_count | 79.0 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
