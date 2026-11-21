## gb300-pg62-A6-c16-r2 (2026-11-21T14:06:27Z)

dsv41-5way gb300-pg62-A6 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2466.53 | 210.28 | 0.179 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 294.08 |
| json | 197.26 |
| narrative | 122.21 |
| prose | 159.55 |
| math | 274.09 |
| reasoning | 207.11 |
| summary | 154.85 |
| format | 273.11 |
| ceiling_count | 310.24 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
