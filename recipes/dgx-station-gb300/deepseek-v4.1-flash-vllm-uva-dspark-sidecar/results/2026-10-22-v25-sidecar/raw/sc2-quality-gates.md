**Verdict: PASS** — every pre-registered gate passed. NOT promoted (James decides).
GATE_RESULT PASS
SUFFIX SC2-PASS

Map rowmap-static-v1.json; C-off mode flag.

| # | gate | result | detail |
|---|---|---|---|
| 0 | valid | PASS | all validity checks |
| 1 | tf_flips | PASS | C vs tf-CTL1 PASS W=2.974 (calib 3.078); C vs A PASS W=3.014; floor A vs tf-CTL1 W=1.866, D vs A W=1.637 |
| 2 | gsm8k_1319 | PASS | C 96.97 (1279/1319) vs A 96.97 (+0.00 pt; McNemar p=1.0; ctok ratio 1.5941237736353229) bar ≥ −1.0 |
| 3 | bfcl | PASS | dev C 92.67 vs A 92.33 (+0.34 pt); heldout C 77.65 vs A 77.88 (-0.23 pt) — bar ≥ −1 pt each |
| 4 | gauntlet | PASS | C 12/12 (A (12, 12), C-off (12, 12)) |
| 5 | needle | PASS | C EXACT 500/500 (near_miss 0, wrong_facts 0, fail 0); A 500/500 — bar 500/500 EXACT |
| 6 | reasoning_effort | PASS | C none=200 low=200 high=200 xhigh=200 max=200 medium=400 | A none=200 low=200 high=200 xhigh=200 max=200 medium=400 (v23 expectation 200×5, medium 400) |
| 7 | image | PASS | A http=200 delta=186 named=True | C http=200 delta=185 named=True answer='The image contains red on the left half and blue on the right half, split by a vertical line.' |
| 8 | perclass | PASS | 20/20 comparisons ≥ −3% |
| 9 | replay | PASS | C r2 480.3 vs mean(A,D) r2 418.9 (+14.7%, bar ≥ −3%); tool turns emitted C r1/r2 15/14 (bar ≥ 4 each) |
| 10 | soak | PASS | 1220 s, 41 iterations, 1312 requests, errors 0, nonzero rc 0, sidecar timeouts Δ 0, wait 0.035 ms/wait; health True |
| 11 | sidecar_timeouts | PASS | C: shm end 0 Δ 0 peer-log 0 hook 0 (watchdog trips: timeout 0, other 0); Coff: shm end 0 Δ 0 peer-log 0 hook 0 (watchdog trips: timeout 0, other 0) — bar 0 everywhere |
| 12 | tool_emission | PASS | C tool-turn emitted per pass [8, 6]/12 (bar >= min(A,D) 5 - 1); C text-turn spurious [5, 4] (bar <= max(A,D) 4 + 1); emit flips vs A 10 (A vs D control 6); errors 0 |

Validity: C: peer mode 2, check 0 (production) OK; C: MEGA_PEER rowmap /w/rowmap-static-v1.json: 40 layers OK; C: decoder SWA bounded replay line (overlay ON) OK; perclass prompts sha1 618898aa (sc2c1) OK; perclass prompts sha1 618898aa (sc2c2) OK; perclass prompts sha1 618898aa (sc2d) OK | C-off valid: True

### Speed and resources (reported, not gated)

| cell | A v23 live | C sidecar | C-off | D v23 restart | C vs mean(A,D) |
|---|---|---|---|---|---|
| RT C1 decode_agg (mean of 3) | 187.8 | 279.2 | 276.7 | 186.3 | +49.3% |
| RT C1 median TTFT s | 0.660 | 0.190 | 0.290 | 0.660 | -71.2% |
| RT C1 acceptance_len | 2.955 | 2.961 | 2.938 | 2.938 | +0.5% |
| knee6 C1 (4-rep mean) | 192.0 | 222.8 | 219.8 | 192.0 | +16.0% |
| knee6 C6 (4-rep mean) | 505.3 | 762.4 | 752.2 | 519.0 | +48.9% |
| knee6 C8 (4-rep mean) | 598.6 | 891.1 | 925.0 | 601.9 | +48.5% |
| knee r2 C1 | 192.1 | 218.6 | — | — | — |
| knee r2 C8 | 600.8 | 925.1 | — | — | — |
| knee r2 C16 | 937.4 | 1414.5 | — | — | — |
| replay r1 agg tok/s | 333.4 | 488.9 | 493.7 | 374.7 | +38.1% |
| replay r2 agg tok/s | 408.2 | 480.3 | 612.1 | 429.5 | +14.7% |
| cold 64K prefill tok/s (N=1) | 23439 | 50119 | — | — | — |
| KV tokens | 2504891 | 2537081 | 2733593 | 2504891 | |
| RTX memory.used MiB (after bind) | | 66213 | 66213 | | |
| power C | GB300 p99 984 W max 988 W sw_power_cap 5.91% | RTX p99 224 W max 229 W | n=3401 | | |
| power Coff | GB300 p99 949 W max 958 W sw_power_cap 4.70% | RTX p99 215 W max 224 W | n=149 | | |
| GB300 wait ms/wait C | | 0.0150 | waits 5665365 | | |
| GB300 wait ms/wait Coff | | 0.0148 | waits 239848 | | |

### Replay ON (C, overlay #58132 + KV-side bounded replay) vs OFF (C-off, mode=flag) — reported, not gated

- replay r2: OFF 612.1 vs ON 480.3 (+27.4%); r1 OFF 493.7 vs ON 488.9
- RT C1: OFF 276.7 vs ON 279.2; TTFT OFF 0.290 s vs ON 0.190 s (Δ +0.100 s)
- TF: C-off vs A PASS W=3.005; C-off vs C W=2.715 (b12x is not bitwise; this is the sidecar same-map floor)
- gauntlet: C (12, 12) C-off (12, 12)
- Recommendation (heuristic, reported): **ship replay OFF (--no-swa-bounded-replay): warm agent replay r2 ≥ +3% over ON, = v23 agent-lane policy**

### Per-class decode wall-clock (GATE: C mean of 2 runs ≥ −3% vs BOTH v23 baseline and D, every class × {C1, C8})

| class | cell | C (run1 / run2) | v23 mean | Δ | D mean | Δ |
|---|---|---|---|---|---|---|
| prose | c1 | 294.8 (294.6 / 295.0) | 232.3 | +26.9% | 232.0 | +27.1% |
| prose | c8 | 1022.6 (1020.3 / 1025.0) | 532.0 | +92.2% | 518.1 | +97.4% |
| structured | c1 | 299.1 (304.5 / 293.7) | 250.3 | +19.5% | 251.7 | +18.9% |
| structured | c8 | 991.7 (1019.6 / 963.7) | 509.4 | +94.7% | 505.0 | +96.4% |
| code | c1 | 408.9 (404.1 / 413.7) | 293.4 | +39.4% | 296.7 | +37.8% |
| code | c8 | 1274.9 (1267.0 / 1282.9) | 642.1 | +98.6% | 663.8 | +92.1% |
| shell_ops | c1 | 413.7 (416.5 / 410.9) | 316.9 | +30.6% | 319.5 | +29.5% |
| shell_ops | c8 | 1325.2 (1322.5 / 1327.8) | 681.4 | +94.5% | 659.6 | +100.9% |
| tool_json | c1 | 367.2 (364.0 / 370.5) | 300.0 | +22.4% | 296.3 | +23.9% |
| tool_json | c8 | 1232.0 (1218.6 / 1245.4) | 684.0 | +80.1% | 682.0 | +80.6% |

## Milo review notes (2026-10-22 00:15 CDT) — not part of the generated verdict
- **GSM8K completion-token ratio 1.59 is one runaway, not a length shift.** Item 1304 on C ran to the 131,072-token cap
  (finish=length, wrong, 305 s); on A the same item stopped at 2,245 tokens and was correct. Excluding it, C mean ≈ 299 vs A ≈ 250
  (p50 175 vs 162, p90 420 vs 369; >4K-token outputs 9 vs 5). Accuracy identical 1279/1319 (C lost 1304, gained another).
  A temp-0 runaway on a daily-driver lane is a real risk (stuck agent turn, ~5 min of GPU): needs a repeat probe (item 1304 ×N
  on sidecar vs v23) before promotion. Not a gate in SC2; flagged.
- **SC1 tool-emission flag did not reproduce:** replay_c C 15/14 vs A 15/15, D 15/15. C-off 13/13. The per-turn temp-0 dump is
  noisy even on v23 (A pass0 9/12 vs pass1 5/12), so gate 12 has low power; replay_c across 4 lanes is the better read.
- **Power:** GB300 sw_power_cap active 5.91% of 3,401 samples on C (4.70% C-off) vs SC6 bar <1% → SC6 bar is failed today.
- Replay OFF vs ON: r2 +27% is one rep (r1 equal) — treat as noise until repeated; TTFT ON 0.19 vs OFF 0.29 s.
