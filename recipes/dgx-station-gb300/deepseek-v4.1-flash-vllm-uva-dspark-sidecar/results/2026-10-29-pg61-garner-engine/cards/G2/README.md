# Card G2 — promotion-tier gates for Pg (Garner tree under our policy + #60945 guard), 2026-10-29

**Verdict: PASS on every quality gate; one speed regression (prefill −8 %).** Not promoted. Still owed: T3, a fail-closed
watchdog for peer3, the #61161 temperature > 0 check on v0.31.0, a reproducible launcher, and the 48 h soak (which
means Pg serving guests — James's call).

Lane dark 14:30–15:01 CDT; h2-REF restored (smoke OK, watchdog armed, isolate off). Pg booted with the guard in the
container (`4c6250af`), 24 components ACTIVE, 0 fallback. Same suite, flags and baselines as the v27 promotion run
(`results/2026-10-26-v27-t2-heldout/` in the sidecar recipe); the M4 drove T0/T2/T1/BFCL over the LAN through a
single-host ACCEPT in the isolate rule.

| check | Pg | reference | verdict |
|---|---|---|---|
| T0 (models / exact word / tool call) | all three OK | — | PASS |
| needle-checksum, 500 requests | **EXACT 500/500**, 0 max_tokens hits | v27 EXACT 500/500 | PASS |
| GSM8K-200 paired (T=0, C4) | **193/200 = 96.5 %**, 0 unparseable, 0 max_tokens | baseline 192/200; flips 0 lost / 1 gained, McNemar p = 1.0; completion tokens +11.8 % | PASS |
| tool-eval, hardmode, 92 scenarios | **91/100, 167/184**, 79 pass, not safety-capped; fails TC-43, 68, 89, 92; TC-85 partial | v27 89/100, 163/184, safety-capped; fails TC-43, 51, 68, 74, 89, 92 | PASS (better) |
| BFCL held-out (live_simple + live_multiple) | **1026/1311 = 78.26 %** (simple 80.23, multiple 77.78), 0 errors | v27 1013/1311 = 77.27 % → **+0.99 pt; paired 5 lost / 18 gained, McNemar exact p = 0.011** | PASS (significantly better) |
| replay_turns ×6 (24-turn Hermes transcript, C1, T=0) | tool turns emitted 7.67/12 mean (7,7,10,8,6,8); text-turn false emits 3.83/12 | h2-REF same window: 8.00/12 (9,8,7,7,9,8); 3.50/12 | within the pass-to-pass spread (6–10); direction slightly negative |
| T1 C1 / C4 × 0 / 32K / 128K (lib synthetic, 30 s) | C1 373.6 / 328.3 / 334.5 · C4 769.1 / 775.8 / 755.7, 0 errors | v27 C1 262.1 / 254.1 / 249.7 · C4 661.1 / 642.5 / 636.7 | **1.29–1.43× C1, 1.16–1.21× C4** |
| prefill, his bench (uncached, 3 reps) | 14K 356 ms (39.9K tok/s) · 57K 1,366 ms (41.6K) · 114K 2,774 ms (41.0K) | h2-REF same day: 324 ms (44.0K) · 1,280 ms (44.4K) · 2,548 ms (44.6K) | **REGRESSION −8 to −10 %** TTFT; matches the 1M needle (36.5 s vs 33.2 s) |

Notes
- T1/BFCL/tool-eval references are v27's promotion run (2026-10-26), the last full T2/BFCL on this lane; h1-REF and
  h2-REF never ran BFCL. h2-REF is ~1 % faster than v27 at C1 (h1 lever), so the T1 ratios are slightly flattering.
- Prefill regression: likely the larger cold set (129/layer on the RTX vs our 89) on the T > 64 path, which still uses
  peer tier v2 (b12x) — peer3 only arms for T ≤ 64. Not measured; a rowmap sweep (h270/h285 under our policy) would
  separate it, at the cost of KV tokens.
- Replay and C1 per-class speed were run once per arm; replay passes vary 6–10/12 within an arm.
