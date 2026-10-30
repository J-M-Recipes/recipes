# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T13:35:14+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 731.3 (702.5-762.0) | 141 (140-143) | 5.39 (5.21-5.62) | 135.9 (135.8-136.0) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 509.8 | — | 3.892 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 591.4 | — | 4.676 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,404 | 175.4 | 3.058 | 60 |
| 32 | 2,382 | 74.4 | 3.043 | 60 |
