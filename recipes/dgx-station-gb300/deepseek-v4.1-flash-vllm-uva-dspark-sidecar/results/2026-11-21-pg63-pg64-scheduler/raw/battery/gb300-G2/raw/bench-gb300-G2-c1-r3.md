## gb300-G2-c1-r3 (2026-11-21T14:38:40Z)

dsv41-5way gb300-G2 c1 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 484.95 | 551.15 | 0.034 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 792.19 |
| json | 564.35 |
| narrative | 310.78 |
| prose | 334.23 |
| math | 680.3 |
| reasoning | 555.74 |
| summary | 373.61 |
| format | 798.0 |
| ceiling_count | 1045.05 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
