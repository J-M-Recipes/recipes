# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T15:16:47+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 722.3 (715.8-752.9) | 129 (129-132) | 5.30 (5.26-5.53) | 135.9 (135.8-136.1) | 3 |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,463 | 182.8 | 3.051 | 60 |
| 32 | 2,664 | 83.2 | 3.028 | 60 |
