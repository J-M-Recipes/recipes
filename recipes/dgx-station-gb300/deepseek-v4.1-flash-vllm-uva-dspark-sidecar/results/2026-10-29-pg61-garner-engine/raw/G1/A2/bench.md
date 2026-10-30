# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T14:49:58+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 447.4 (440.6-452.8) | 133 (133-134) | 4.69 (4.61-4.73) | 95.8 (95.3-96.0) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 335.7 | — | 3.516 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 400.9 | — | 4.336 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 1,124 | 140.6 | 3.067 | 60 |
| 32 | 1,463 | 45.7 | 3.076 | 60 |
