# Card G4 — T3 promotion bench, Pg61 (booted by the launcher) vs h2-REF same window (2026-10-29)

**Verdict: PASS on all six pre-registered bars** (PREREG.md). Lane dark 6:08–9:03pm CDT (MMLU-Pro took 50 / 65 min per
arm, not the ~10 assumed; ETA slipped from 7:35 to 9:05pm). h2-REF serving since 7:34pm (smoke OK, watchdog armed);
isolate held until the control suite finished, then G5 started at 9:03pm.

| bar | Pg61 | h2-REF (A), same window | verdict |
|---|---|---|---|
| 5 launcher reproduces G3 | fresh `build` → `up`: hashes OK, **`docker inspect` Args identical to G3**, 24 ACTIVE, KV 11.70M | — | PASS |
| 1 needle-checksum 500 | EXACT 500/500 | EXACT 500/500 | PASS |
| 2 GSM8K-1319 | 96.97 % (1279) · vs v25m p = 0.73 (3 lost / 5 gained) | 96.89 % (1278) · paired 4 lost / 5 gained, p = 1.0 | PASS |
| 3 MMLU-Pro-1000 | **87.7 % (877)**, 14 hit 131K max_tokens | 85.3 % (853), 20 hit max_tokens · paired **10 lost / 34 gained, p < 0.001** | PASS (better) |
| 4 T1 C1 (0/32K/128K) | 325 / 322 / 321 tok/s | 247 / 266 / 250 | PASS: 1.31× / 1.21× / 1.28× |
| 4 T1 C4 | 774 / 757 / 804 | 690 / 679 / 679 | PASS: 1.12× / 1.11× / 1.18× |
| T1 C8 (no bar) | 1,288 / 1,259 / 1,262 | 1,124 / 1,127 / 1,124 | 1.15× / 1.12× / 1.12× |
| 6 watchdog / survival | 0 trips, server up | — | PASS |
| tool-eval ×2 (no bar) | 89 (163/184) · 91 (168) | 89 (163, safety-capped) · 93 (171) | same band; one run cannot rank |

Notes
- MMLU-Pro gap is not answer length: median completion 313 vs 310 tokens; neither container sets a reasoning-effort
  default. Runaway generations to the 131K cap (14 vs 20) explain at most 6 of the net +24. Cause not investigated.
- T1 runs at the server's default temperature (not 0). C1 acceptance Pg61 0.30/0.31/0.31 ≈ h2-REF 0.30/0.35/0.31
  (both carry #61161); Pg (pre-port, G2) read 0.357/0.313/0.331 → G5 tests whether the port lowers T > 0 acceptance.
  **G5 answer: it does not** (per-class C1, mean of 3 at T = 1.0: Pg61/Pg = 1.012). The filler C1 drop was noise.
- Not exercised: temperature > 0 quality; decode beyond 128K here (1M needle in G1f).
