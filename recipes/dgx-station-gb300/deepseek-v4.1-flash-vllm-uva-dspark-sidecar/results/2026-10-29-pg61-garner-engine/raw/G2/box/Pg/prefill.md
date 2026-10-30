# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-10-29T19:39:11+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14220 (14212-14262) | 356 (348-371) | 39854 (38470-40857) |
| 56794 (56789-56818) | 1366 (1349-1383) | 41577 (41087-42094) |
| 113658 (113578-113680) | 2774 (2773-2777) | 40987 (40903-40989) |
