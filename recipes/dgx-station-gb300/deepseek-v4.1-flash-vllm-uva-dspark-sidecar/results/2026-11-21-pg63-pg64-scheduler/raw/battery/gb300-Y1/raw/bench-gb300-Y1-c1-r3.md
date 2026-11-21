## gb300-Y1-c1-r3 (2026-11-21T16:22:43Z)

dsv41-5way gb300-Y1 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 480.22 | 547.07 | 0.035 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 792.6 |
| json | 551.19 |
| narrative | 311.56 |
| prose | 334.77 |
| math | 688.9 |
| reasoning | 558.23 |
| summary | 344.64 |
| format | 794.69 |
| ceiling_count | 1018.66 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
