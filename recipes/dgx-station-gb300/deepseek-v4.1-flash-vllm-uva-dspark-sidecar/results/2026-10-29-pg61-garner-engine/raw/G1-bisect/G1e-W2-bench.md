# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T17:11:21+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 660.3 (655.3-673.7) | 139 (135-140) | 4.94 (4.89-5.05) | 133.6 (133.6-134.0) | 3 |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,454 | 181.8 | 3.057 | 60 |
| 32 | 2,599 | 81.2 | 3.065 | 60 |
