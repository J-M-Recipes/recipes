## gb300-Y2-c16-r2 (2026-11-21T16:44:16Z)

dsv41-5way gb300-Y2 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2454.55 | 214.02 | 0.232 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 299.01 |
| json | 191.78 |
| narrative | 116.84 |
| prose | 163.86 |
| math | 279.44 |
| reasoning | 213.3 |
| summary | 150.91 |
| format | 297.05 |
| ceiling_count | 311.64 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
