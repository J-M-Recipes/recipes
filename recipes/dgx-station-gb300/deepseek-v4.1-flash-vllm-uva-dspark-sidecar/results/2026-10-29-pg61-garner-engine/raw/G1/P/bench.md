# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T14:20:50+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 636.3 (631.8-655.0) | 128 (127-128) | 5.37 (5.29-5.49) | 119.6 (119.0-119.6) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 451.2 | — | 3.776 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 548.5 | — | 4.706 | — | window |
