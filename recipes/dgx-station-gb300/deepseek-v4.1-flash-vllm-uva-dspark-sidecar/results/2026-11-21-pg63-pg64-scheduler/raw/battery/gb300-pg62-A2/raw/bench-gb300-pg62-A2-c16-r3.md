## gb300-pg62-A2-c16-r3 (2026-11-21T11:53:29Z)

dsv41-5way gb300-pg62-A2 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2487.66 | 211.67 | 0.175 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.09 |
| json | 199.22 |
| narrative | 117.81 |
| prose | 167.18 |
| math | 270.26 |
| reasoning | 218.66 |
| summary | 154.88 |
| format | 272.24 |
| ceiling_count | 310.52 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
