# Card F3 verdict: one-user decode, Pg67c vs Flare gb300 core d683cc1a (runtime 7db06e0c)

**Winner: pg67c** (primary pg67c, secondary pg67c; a lead under 3 % counts as a tie)

| instrument | Pg67c (GB300 + RTX sidecar) | Flare (one GB300) | Flare / Pg67c |
|---|--:|--:|--:|
| llm-inference-bench C1 ctx0, median of 3/3 runs | 389.4 ([403.1, 389.4, 380.6]) | 322.7 ([333.9, 322.7, 320.5]) | 0.83 |
| per-class C1, mean of 3/3 seeds | 498.9 | 366.3 | 0.73 |
| per-class C1, natural length | 541.1 | 392.5 | |

## llm-inference-bench full table (one run per arm; aggregate tok/s)

| arm | C1 0 | C8 0 | C32 0 | C1 16K | C8 16K | C32 16K |
|---|--:|--:|--:|--:|--:|--:|
| Pg67c | 403 | 1,267 | 2,405 | 365 | 1,272 | 2,407 |
| Flare | 334 | 1,137 | 2,379 | 347 | 1,146 | 2,338 |

## Per-class C1 by class

| arm | prose | structured | code | shell_ops | tool_json |
|---|--:|--:|--:|--:|--:|
| Pg67c | 406.1 | 447.8 | 555.6 | 574.9 | 509.9 |
| Flare | 318.9 | 341.8 | 387.1 | 392.1 | 391.6 |
