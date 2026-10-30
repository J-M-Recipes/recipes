# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T14:34:28+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 706.1 (682.7-714.3) | 133 (132-133) | 5.39 (5.22-5.44) | 131.0 (130.8-131.5) | 10 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 484.9 | — | 3.766 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 582.3 | — | 4.708 | — | window |
