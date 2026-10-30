# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T22:21:15+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 648.0 (641.4-660.7) | 132 (131-133) | 5.44 (5.38-5.55) | 119.3 (119.1-119.5) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 459.5 | — | 3.85 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 545.1 | — | 4.651 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,440 | 179.9 | 3.045 | 60 |
| 32 | 2,604 | 81.4 | 3.054 | 60 |
