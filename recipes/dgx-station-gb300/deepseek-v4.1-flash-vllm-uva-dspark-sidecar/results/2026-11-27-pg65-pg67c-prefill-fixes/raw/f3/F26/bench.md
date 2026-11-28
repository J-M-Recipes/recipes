# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-28T04:04:05+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 484.3 (467.9-510.4) | 118 (118-119) | n/a | n/a | 3 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 0.0 | — | n/a | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 0.0 | — | n/a | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 0 | 0.0 | n/a | 60 |
| 32 | 0 | 0.0 | n/a | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14224 (14216-14226) | 225 (223-225) | 63190 (63184-63657) |
| 56868 (56837-56878) | 739 (736-739) | 76969 (76935-77255) |
| 113658 (113643-113665) | 1463 (1461-1468) | 77689 (77411-77790) |

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 194/200 (97.0 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
