**Verdict: PASS** — every pre-registered gate passed. NOT promoted (James decides).
GATE_RESULT PASS
SUFFIX P2f-PASS

### Gate summary

| gate | result | detail |
|---|---|---|
| perclass | PASS | 20/20 comparisons pass |
| gsm8k | PASS | C 95.5 vs A 97.0 (-1.5 pt; McNemar p=0.25) bar ≥ −1.5 |
| bfcl_dev | PASS | C 92.50 vs A 92.33 (+0.17 pt) bar ≥ −1 |
| bfcl_heldout | PASS | C 77.65 vs A 77.65 (+0.00 pt) bar ≥ −1 |
| gauntlet | PASS | C 12/12 |
| rt_c1 | PASS | C 197.8 vs CTL 186.4 (+6.1%) bar ≥ −2% |
| knee6_c8 | PASS | C 656.5 vs CTL 605.2 (+8.5%) bar ≥ −2% (4 reps/arm) |

Validity: rehome done layers=40 OK; no 'Decoder SWA bounded replay:' line OK; MegaAttn backend line OK; prompts sha1 618898aa (v23) OK; prompts sha1 618898aa (candC) OK; prompts sha1 618898aa (v23post) OK

### Per-class decode wall-clock (GATE: every class x {C1,C8} >= bar vs BOTH controls; bar = ctl mean x (1 − max(2%, 1.5 x ctl spread)))

| class | cell | candC mean | v23 (18:29) mean / spread | bar | Δ | verdict | v23post mean / spread | bar | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| prose | c1 | 241.6 | 232.3 / 0.3% | 227.7 | +4.0% | PASS | 231.4 / 0.4% | 226.7 | +4.4% | PASS |
| prose | c8 | 564.8 | 532.0 / 2.5% | 511.9 | +6.2% | PASS | 519.3 / 1.9% | 504.8 | +8.8% | PASS |
| structured | c1 | 255.5 | 250.3 / 0.8% | 245.3 | +2.1% | PASS | 254.3 / 1.6% | 248.3 | +0.5% | PASS |
| structured | c8 | 573.8 | 509.4 / 2.7% | 488.6 | +12.6% | PASS | 522.2 / 0.8% | 511.8 | +9.9% | PASS |
| code | c1 | 312.6 | 293.4 / 3.3% | 278.8 | +6.6% | PASS | 296.1 / 1.6% | 289.2 | +5.6% | PASS |
| code | c8 | 673.2 | 642.1 / 1.2% | 629.2 | +4.8% | PASS | 642.8 / 4.6% | 598.7 | +4.7% | PASS |
| shell_ops | c1 | 331.6 | 316.9 / 5.0% | 292.9 | +4.6% | PASS | 317.0 / 2.9% | 303.4 | +4.6% | PASS |
| shell_ops | c8 | 739.1 | 681.4 / 3.3% | 647.5 | +8.5% | PASS | 657.7 / 0.7% | 644.6 | +12.4% | PASS |
| tool_json | c1 | 316.9 | 300.0 / 1.6% | 293.0 | +5.6% | PASS | 286.2 / 0.1% | 280.5 | +10.7% | PASS |
| tool_json | c8 | 767.1 | 684.0 / 3.1% | 652.0 | +12.1% | PASS | 721.8 / 1.7% | 703.1 | +6.3% | PASS |

Per-class gate: **PASS** (20/20 comparisons pass).

### Per-class acceptance (/metrics, mean of r1/r2) and the acceptance-tax vs kernel-gain split
tokens/step ≈ 1 + 5a (k=5 at C1 and C8). step-speed ratio = decode ratio ÷ tokens/step ratio (>1 = the engine's step is faster at equal acceptance).

| class | cell | accept candC | v23 | v23post | Δ pts vs v23 / v23post | decode ratio vs mean(v23, v23post) | tokens/step ratio | step-speed ratio (kernel gain) |
|---|---|---|---|---|---|---|---|---|
| prose | c1 | 40.5 | 41.8 | 41.5 | -1.3 / -1.0 | 1.042 | 0.982 | 1.061 |
| prose | c8 | 43.1 | 42.2 | 41.0 | +0.9 / +2.1 | 1.074 | 1.025 | 1.048 |
| structured | c1 | 42.9 | 45.5 | 46.5 | -2.6 / -3.6 | 1.013 | 0.953 | 1.062 |
| structured | c8 | 43.2 | 42.8 | 43.5 | +0.4 / -0.2 | 1.112 | 1.001 | 1.111 |
| code | c1 | 66.5 | 66.6 | 66.9 | -0.1 / -0.4 | 1.061 | 0.997 | 1.064 |
| code | c8 | 67.7 | 67.6 | 69.3 | +0.1 / -1.6 | 1.048 | 0.991 | 1.057 |
| shell_ops | c1 | 67.1 | 68.4 | 67.9 | -1.4 / -0.8 | 1.046 | 0.988 | 1.059 |
| shell_ops | c8 | 67.2 | 66.6 | 66.5 | +0.6 / +0.7 | 1.104 | 1.007 | 1.096 |
| tool_json | c1 | 59.6 | 59.6 | 56.3 | +0.0 / +3.4 | 1.081 | 1.022 | 1.058 |
| tool_json | c8 | 59.5 | 57.4 | 59.1 | +2.1 / +0.4 | 1.091 | 1.016 | 1.074 |

TRIGGER (report, not gate) — acceptance moves > 2 pts: prose/c8 vs v23post: 43.1 vs 41.0 (+2.1 pts); structured/c1 vs v23: 42.9 vs 45.5 (-2.6 pts); structured/c1 vs v23post: 42.9 vs 46.5 (-3.6 pts); tool_json/c1 vs v23post: 59.6 vs 56.3 (+3.4 pts); tool_json/c8 vs v23: 59.5 vs 57.4 (+2.1 pts)

Window drift (v23post vs v23 18:29, |Δ| > max of the two same-config spreads): prose/c1 -0.4% (spreads 0.3/0.4%); structured/c1 +1.6% (spreads 0.8/1.6%); shell_ops/c8 -3.5% (spreads 3.3/0.7%); tool_json/c1 -4.6% (spreads 1.6/0.1%); tool_json/c8 +5.5% (spreads 3.1/1.7%)

### Fidelity (control = A, same window; D is report-only)

| cell | A v23 live | C config C | D v23 restart | C − A | bar | verdict |
|---|---|---|---|---|---|---|
| GSM8K-200 accuracy % | 97.0 (194/200) | 95.5 (191/200) | 96.5 (193/200) | -1.5 | ≥ −1.5 | PASS |
| BFCL dev (simple_python+multiple, 600) % | 92.33 (554/600) | 92.50 (555/600) | 92.67 (556/600) | +0.17 | ≥ −1 | PASS |
| BFCL held-out (live_simple+live_multiple) % | 77.65 (1018/1311) | 77.65 (1018/1311) | 77.88 (1021/1311) | +0.00 | ≥ −1 | PASS |
| gauntlet | 12/12 | 12/12 | 12/12 | | 12/12 | PASS |

GSM8K: C vs A McNemar p = 0.25, completion tokens mean C 293.515 (A vs Card D baseline: Δ 0.5000000000000004 pp, p=1.0).
GSM8K D vs A (same image, two boots): Δ -0.5000000000000004 pp, p=1.0 — the same-config noise floor for this instrument.

### Speed (CTL = mean of A and D)

| cell | A v23 live | C config C | D v23 restart | C vs CTL | gate |
|---|---|---|---|---|---|
| RT C1 decode_agg (mean of 3) | 186.1 | 197.8 | 186.8 | +6.1% | PASS |
| RT C1 median TTFT s (≈ TTFT 8K) | 0.643 | 0.430 | 0.647 | -33.3% |  |
| RT C1 acceptance_len | 2.910 | 2.940 | 2.942 | +0.5% |  |
| knee6 C1 (mean of 4 reps) | 192.1 | 200.4 | 192.2 | +4.3% |  |
| knee6 C6 (mean of 4 reps) | 530.0 | 557.5 | 525.3 | +5.7% |  |
| knee6 C8 (mean of 4 reps) | 596.5 | 656.5 | 614.0 | +8.5% | PASS |
| knee6 C8 raw reps | [615.7, 569.2, 622.8, 578.2] | [645.9, 647.3, 649.5, 683.4] | [622.7, 603.8, 625.9, 603.7] | | |
| knee r2 C1 | 192.2 | 200.3 | 192.4 | +4.2% |  |
| knee r2 C8 | 618.5 | 647.8 | 600.6 | +6.3% |  |
| knee r2 C16 | 944.4 | 1074.8 | 944.5 | +13.8% |  |
| cold 64K prefill tok/s (N=1) | 24094 | 30518 | 24006 | +26.9% |  |
| KV tokens | 2,504,891 (v23) | 3,878,024 | | | |
| bind s | live | 335 | 295 | | |
