RESULT PASS

| gated cell | A = v22 REF (live) | B = S0 (restart) | Δ B vs A | bar ≥ −2.0% |
|---|---|---|---|---|
| knee6 r2 C8 | 597.0 | 604.8 | 1.31% | PASS |
| knee6 r2 C6 | 535.8 | 529.6 | -1.16% | PASS |
| RT C1 (kbench, mean of 3) | 187.8 | 188.2 | 0.23% | PASS |

| informational | A r1 | A r2 | B r1 | B r2 | Δ r2 |
|---|---|---|---|---|---|
| knee6 C6 | 534.0 | 535.8 | 530.3 | 529.6 | -1.16% |
| knee6 C8 | 624.4 | 597.0 | 590.3 | 604.8 | 1.31% |
| knee6 C16 | 952.8 | 954.4 | 752.9 | 945.5 | -0.93% |
| knee6 C1 | 192.7 | 192.5 | 192.5 | 192.5 | -0.01% |
| knee.sh C8 (S0 caveat instrument) | 596.7 | 629.3 | 623.6 | 598.5 | -4.89% |
| knee.sh C16 | 956.6 | 956.2 | 946.0 | 946.4 | -1.03% |
| knee.sh C1 | 192.4 | 192.5 | 192.3 | 192.5 | -0.03% |
| RT C1 runs | 188.3 / 185.6 / 189.4 | | 190.8 / 189.1 / 184.8 | | |

host pinned (/dev/zero Size sum): A 327.96 GiB · B 252.79 GiB
MemAvailable: A live 123.99 GiB · B steady 196.23 GiB
B hook guard host_avail: L9 197.69 GiB · min 153.55 GiB (guard 10 GiB)
B KV tokens 2,504,891 · autotune Loaded 189 configs, Saved lines 1
