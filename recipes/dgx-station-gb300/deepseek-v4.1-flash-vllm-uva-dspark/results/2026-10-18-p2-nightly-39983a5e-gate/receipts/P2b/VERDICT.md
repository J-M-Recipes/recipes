**Verdict: candidate none** — neither B nor C meets the pre-registered P2b bars — P2 closes
SUFFIX_B P2b-FAIL
SUFFIX_C FAIL
CANDIDATE none

Bars (CTL = mean of A and D):
- B (mechanism valid=True): FL replay r2 ≥ −2%: MISS [463.6 vs 495.5 (-6.4%)]; fixture_x3 per-class acceptance within 2 pts: MISS [prose +0.2, structured -1.7, code -4.7, shell_ops +0.3, tool_json -1.7]; RT C1 ≥ −2%: PASS [197.3 vs 187.8 (5.1%)]
- C (mechanism valid=True): FL replay r2 ≥ −2%: PASS [548.4 vs 495.5 (10.7%)]; fixture_x3 per-class acceptance within 2 pts: MISS [prose +0.2, structured -1.7, code -4.7, shell_ops +0.3, tool_json +2.5]; RT C1 ≥ −2%: PASS [198.2 vs 187.8 (5.5%)]; TTFT 8K ≤ 0.45 s: PASS [0.430 s (CTL 0.645)]

| arm | boot s | KV tok | replay line | rehome | tune window | Preloaded | store | RT C1 (runs) | RT acc_len | TTFT 8K s | std r1 tok/s (tok) | std r2 tok/s (tok) | FL r1 tok/s | FL r2 tok/s | FL r2 TPOT mean/med ms | FL r1 TTFT cold/warm s | FL accept r2 | cold 64K TTFT s / tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | live | 2,504,891 | no | yes | 16:24:05→16:24:10 | — | 210 | 187.9 (188.2, 190.4, 185.2) | 2.949 | 0.640 | 420.1 (6044) | 440.7 (6754) | 432.0 | 509.4 | 7.11/6.80 | 0.309/0.116 | 0.567 | 2.16, 2.17 s / 24,547 |
| B | 335 | 3,455,569 | yes | yes | 17:20:44→17:20:52 | 210/210 | 210 | 197.3 (195.5, 197.2, 199.3) | 2.944 | 0.280 | 229.7 (5644) | 382.9 (6430) | 476.0 | 463.6 | 7.65/7.50 | 0.212/0.208 | 0.550 | 1.16, 1.13 s / 46,318 |
| C | 526 | 3,878,024 | no | yes | 17:32:59→17:33:07 | 210/210 | 210 | 198.2 (202.6, 200.4, 191.6) | 2.938 | 0.430 | 194.5 (4973) | 419.0 (5175) | 445.3 | 548.4 | 6.62/6.00 | 0.335/0.199 | 0.565 | 1.78, 1.72 s / 30,362 |
| D | 295 | 2,504,891 | no | yes | 17:41:44→17:41:49 | — | 210 | 187.8 (186.8, 191.3, 185.3) | 2.977 | 0.650 | 272.4 (6097) | 412.9 (5546) | 442.1 | 481.6 | 7.57/7.40 | 0.310/0.108 | 0.518 | 2.23, 2.21 s / 23,981 |

Per-class acceptance % — agent_fixture_x3 (8 prompts/class, token-weighted; per-prompt mean±sd in brackets):
| arm | prose | structured | code | shell_ops | tool_json |
|---|---|---|---|---|---|
| A | 24.7 [25±4] | 46.8 [53±21] | 78.4 [82±11] | 77.0 [79±11] | 83.1 [87±13] |
| B | 24.9 [25±4] | 45.1 [50±20] | 73.6 [79±11] | 77.4 [78±12] | 82.9 [86±12] |
| C | 24.9 [25±4] | 45.1 [50±20] | 73.6 [79±11] | 77.4 [78±12] | 87.1 [88±10] |
| D | 24.7 [25±4] | 46.8 [53±21] | 78.4 [82±11] | 77.0 [79±11] | 86.1 [87±10] |
| CTL | 24.7 | 46.8 | 78.4 | 77.0 | 84.6 |

Per-class acceptance % — standard agent_fixture_o.sh × 3 CAT_ORDER rotations (mean, range):
| arm | prose | structured | code | shell_ops | tool_json |
|---|---|---|---|---|---|
| A | 25.2 (±0.0, n=3) | 48.9 (±0.0, n=3) | 69.6 (±0.0, n=3) | 80.0 (±0.0, n=3) | 73.3 (±0.0, n=3) |
| B | 30.6 (±0.0, n=3) | 42.7 (±0.0, n=3) | 62.7 (±0.0, n=3) | 80.0 (±0.0, n=3) | 69.2 (±0.5, n=3) |
| C | 30.6 (±0.0, n=3) | 42.7 (±0.0, n=3) | 62.7 (±0.0, n=3) | 80.0 (±0.0, n=3) | 79.9 (±0.2, n=3) |
| D | 25.2 (±0.0, n=3) | 48.9 (±0.0, n=3) | 69.6 (±0.0, n=3) | 80.0 (±0.0, n=3) | 73.3 (±0.0, n=3) |

Trajectory identity on agent_fixture_x3 (prompts with byte-identical output vs arm A, per class of 8):
- B vs A: prose 0/8, structured 2/8, code 4/8, shell_ops 6/8, tool_json 0/8
- C vs A: prose 0/8, structured 2/8, code 4/8, shell_ops 6/8, tool_json 0/8
- D vs A: prose 8/8, structured 8/8, code 8/8, shell_ops 8/8, tool_json 0/8

Diagnostics:
- B: std replay r1 -33.7% r2 -10.3% vs CTL; r2 tokens 6430 vs CTL 6,150 (4.6%); FL r1 8.9% FL r2 -6.4%; tool turns/emitted r2 24/14; std accept r2 0.594 FL accept r2 0.55
- C: std replay r1 -43.8% r2 -1.8% vs CTL; r2 tokens 5175 vs CTL 6,150 (-15.9%); FL r1 1.9% FL r2 10.7%; tool turns/emitted r2 24/17; std accept r2 0.568 FL accept r2 0.565
- A: std r2 tool turns/emitted 24/13; std accept r2 0.581; FL accept r2 0.567
- D: std r2 tool turns/emitted 24/15; std accept r2 0.582; FL accept r2 0.518
