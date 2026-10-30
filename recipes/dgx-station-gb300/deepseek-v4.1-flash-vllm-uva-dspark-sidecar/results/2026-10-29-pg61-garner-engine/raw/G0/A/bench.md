# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T13:12:45+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 435.4 (430.0-443.0) | 137 (135-140) | 4.57 (4.51-4.66) | 95.1 (95.0-95.3) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 339.6 | — | 3.683 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 403.9 | — | 4.401 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,147 | 143.4 | 3.042 | 60 |
| 32 | 1,561 | 48.8 | 3.038 | 60 |
