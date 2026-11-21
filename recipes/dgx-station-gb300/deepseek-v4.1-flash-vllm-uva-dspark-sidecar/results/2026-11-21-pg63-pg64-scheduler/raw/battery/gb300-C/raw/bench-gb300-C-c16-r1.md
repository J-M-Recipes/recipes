## gb300-C-c16-r1 (2026-11-21T11:22:08Z)

dsv41-5way gb300-C c16 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 1922.49 | 208.34 | 0.546 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 294.71 |
| json | 189.16 |
| narrative | 121.53 |
| prose | 162.14 |
| math | 279.3 |
| reasoning | 219.54 |
| summary | 142.24 |
| format | 258.06 |
| ceiling_count | 305.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
