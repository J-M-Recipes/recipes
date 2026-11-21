## gb300-pg63-K1-c32 (2026-11-21T16:08:31Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C32 | 3448.32 | 149.03 | 0.282 |

### Per-stream tok/s by category

| category | C32 |
|---|---|
| coding | 203.14 |
| json | 140.5 |
| narrative | 81.15 |
| prose | 116.85 |
| math | 193.5 |
| reasoning | 153.77 |
| summary | 104.39 |
| format | 198.98 |
| ceiling_count | 216.04 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
