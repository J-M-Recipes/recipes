# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T17:59:17+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 632.5 (624.4-664.2) | 134 (132-135) | 5.28 (5.23-5.59) | 119.4 (119.3-119.5) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 450.3 | — | 3.815 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 546.7 | — | 4.767 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,393 | 174.2 | 3.046 | 60 |
| 32 | 2,586 | 80.8 | 3.037 | 60 |
