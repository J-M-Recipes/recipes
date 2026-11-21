## gb300-B-c1-r1 (2026-11-21T11:06:54Z)

dsv41-5way gb300-B c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 445.5 | 537.12 | 0.167 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 826.47 |
| json | 588.41 |
| narrative | 310.55 |
| prose | 334.11 |
| math | 603.53 |
| reasoning | 569.9 |
| summary | 369.8 |
| format | 694.16 |
| ceiling_count | 1026.74 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
