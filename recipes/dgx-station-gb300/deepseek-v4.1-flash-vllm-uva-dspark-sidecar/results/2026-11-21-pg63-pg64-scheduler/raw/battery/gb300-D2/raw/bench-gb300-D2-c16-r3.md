## gb300-D2-c16-r3 (2026-11-21T12:26:04Z)

dsv41-5way gb300-D2 c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2444.8 | 211.22 | 0.187 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.6 |
| json | 194.16 |
| narrative | 122.51 |
| prose | 165.21 |
| math | 267.45 |
| reasoning | 213.24 |
| summary | 155.49 |
| format | 278.13 |
| ceiling_count | 311.21 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
