## gb300-C-c16-r3 (2026-11-21T11:22:31Z)

dsv41-5way gb300-C c16 rep 3/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2422.47 | 205.12 | 0.175 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 287.68 |
| json | 180.62 |
| narrative | 118.24 |
| prose | 161.77 |
| math | 270.64 |
| reasoning | 212.14 |
| summary | 148.61 |
| format | 261.25 |
| ceiling_count | 312.23 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
