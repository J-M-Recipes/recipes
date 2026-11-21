## gb300-pg62-A7-c16-r2 (2026-11-21T14:55:03Z)

dsv41-5way gb300-pg62-A7 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2405.96 | 210.63 | 0.241 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 293.8 |
| json | 195.37 |
| narrative | 118.75 |
| prose | 163.92 |
| math | 273.68 |
| reasoning | 214.8 |
| summary | 151.48 |
| format | 273.21 |
| ceiling_count | 300.66 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
