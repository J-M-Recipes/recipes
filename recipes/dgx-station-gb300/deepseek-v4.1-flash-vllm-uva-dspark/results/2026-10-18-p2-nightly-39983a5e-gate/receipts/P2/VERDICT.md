SUFFIX FAIL
| arm | RT C1 (acc) | C4 | C16 | C32 out | TTFT 8K (RT C1) | cold 8K TTFT / tok/s | cold 64K TTFT / tok/s | knee r2 C8 / knee6 C8 | replay r1/r2 | BFCL dev | gauntlet | KV tok | pinned host GiB | MemAvail GiB | hook L9 host_avail | boot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTL1 | 189.5 (2.98) | 327.9 | 532.0 | 502.2 | 0.650 | 0.35 / 19096 | 2.19 / 24206 | 596.5 / 622.6 | 343/415 | 92.3 | 12/12 | 2504891 | 252.8 | 194.6 | 197.7 | – |
| CAND | 199.4 (2.98) | 344.7 | 602.6 | 588.1 | 0.283 | 0.16 / 42398 | 1.70 / 38019 | 661.7 / 660.4 | 334/372 | 92.3 | 12/12 | 3455569 | 252.8 | 188.7 | 195.3 | 1240s |
| CTL2 | 186.8 (2.94) | 330.4 | 531.7 | 506.4 | 0.643 | 0.36 / 18915 | 2.17 / 24463 | 590.2 / 593.2 | 323/420 | 92.5 | 12/12 | 2504891 | 252.8 | 194.7 | 198.0 | 304s |

| arm | fixture accept % (prose / structured / code / shell_ops / tool_json) | replay boot line | rehome | autotune: Loaded / Saved / Preloaded / starts / tune window / store entries |
|---|---|---|---|---|
| CTL1 | 25.2 / 48.9 / 69.6 / 80.0 / 77.3 | absent | yes (layers=40 HBM_expert=206.61GiB pinned_expert=62.33GiB hbm_free=23.08GiB host_avail=197.69GiB) | 189 / 1 / / / 1 / 15:17:47->15:17:52 / 0 |
| CAND | 30.6 / 42.7 / 62.7 / 80.0 / 72.1 | yes | yes (layers=40 HBM_expert=206.61GiB pinned_expert=62.33GiB hbm_free=23.00GiB host_avail=195.30GiB) | – / 0 / / / 1 / 15:59:25->16:10:48 / 210 |
| CTL2 | 25.2 / 48.9 / 69.6 / 80.0 / 77.3 | absent | yes (layers=40 HBM_expert=206.61GiB pinned_expert=62.33GiB hbm_free=23.08GiB host_avail=198.02GiB) | 189 / 1 / / / 1 / 16:24:05->16:24:10 / 210 |

**Verdict: FAIL** — misses: replay r2 >= -2%, fixture accept per class within 2 pts
- TTFT 8K <= 0.45 s: PASS — 0.283 s (CTL mean 0.647)
- cold 64K prefill tok/s >= +30%: PASS — x1.562 (38019 vs CTL 24334 tok/s; TTFT 1.70 vs 2.18 s)
- RT C1 >= -2%: PASS — 6.0% (199.4 vs 188.1)
- replay r2 >= -2%: MISS — -10.9% (372 vs 417)
- flip gate (margin-weighted, ref CTL1): PASS — GATE PASS hi_agent_tool=0 hi_prose_pct=0.011 W=3.110 W_calib=3.078 skipped_docs=0
- gauntlet 12/12: PASS — 12/12
- fixture accept per class within 2 pts: MISS — prose 5.4 pt (CTL1-CTL2 spread 0.0), structured -6.2 pt (CTL1-CTL2 spread 0.0), code -6.9 pt (CTL1-CTL2 spread 0.0), shell_ops 0.0 pt (CTL1-CTL2 spread 0.0), tool_json -5.2 pt (CTL1-CTL2 spread 0.0)
- RT C1 acceptance_len within 0.10: PASS — 2.980 vs 2.958
- BFCL dev >= CTL mean - 1 pt: PASS — 92.3 vs 92.4
- record only: C4 4.7%, C16 13.3%, C32 out 16.6%, knee r2 C8 11.5%, knee6 C8 8.6% (1 rep — not gated), cold 8K TTFT 0.16 vs 0.35 s, KV 3455569 vs CTL 2504891.0
- flip-gate lines: CAND_vs_CTL1: GATE PASS hi_agent_tool=0 hi_prose_pct=0.011 W=3.110 W_calib=3.078 skipped_docs=0 | CAND_vs_Rref: GATE PASS hi_agent_tool=0 hi_prose_pct=0.009 W=3.106 W_calib=3.077 skipped_docs=0 | CTL1_vs_Rref: GATE PASS hi_agent_tool=0 hi_prose_pct=0.011 W=1.878 W_calib=3.077 skipped_docs=0 | CTL2_vs_CTL1: GATE PASS hi_agent_tool=0 hi_prose_pct=0.005 W=1.551 W_calib=3.078 skipped_docs=0 | CAND_vs_CTL2: GATE PASS hi_agent_tool=0 hi_prose_pct=0.008 W=3.165 W_calib=3.276 skipped_docs=0
