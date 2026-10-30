# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T13:55:33+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 446.9 (442.2-448.7) | 135 (135-136) | 4.71 (4.68-4.75) | 94.7 (94.5-94.8) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 334.2 | — | 3.565 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 397.4 | — | 4.364 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,120 | 139.9 | 3.061 | 60 |
| 32 | 1,490 | 46.6 | 3.047 | 60 |
