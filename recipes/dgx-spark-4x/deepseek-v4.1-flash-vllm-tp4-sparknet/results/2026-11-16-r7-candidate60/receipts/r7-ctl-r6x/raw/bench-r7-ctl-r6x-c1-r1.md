## r7-ctl-r6x-c1-r1 (2026-11-17T00:49:33Z)

dsv41-5way r7-ctl-r6x c1 rep 1/3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 91.98 | 103.24 | 0.164 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 132.2 |
| json | 103.26 |
| narrative | 61.15 |
| prose | 77.44 |
| math | 126.41 |
| reasoning | 100.6 |
| summary | 78.1 |
| format | 146.77 |
| ceiling_count | 160.16 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
