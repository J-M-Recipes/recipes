## gb300-G1-c16-r2 (2026-11-21T14:22:52Z)

dsv41-5way gb300-G1 c16 rep 2/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C16 | 2417.15 | 221.52 | 0.256 |

### Per-stream tok/s by category

| category | C16 |
|---|---|
| coding | 292.33 |
| json | 195.05 |
| narrative | 117.94 |
| prose | 153.06 |
| math | 304.44 |
| reasoning | 215.05 |
| summary | 149.68 |
| format | 344.61 |
| ceiling_count | 311.77 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
