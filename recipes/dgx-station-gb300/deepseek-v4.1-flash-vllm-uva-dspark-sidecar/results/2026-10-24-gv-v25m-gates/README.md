# 2026-10-24 — Card GV: fidelity gates for v25m (v25 sidecar + rowmap-mixed-ar-v1.json)

Reference = SC2's C arm (v25 static map, 2026-10-21, same image/hook/b12x). Live v25m lane (V), a check-mode boot (K) for the
bring-up cosine gate, then a fresh production boot (P) that started the 48 h soak (06:14 10-24 → 06:14 10-26 CDT).

**Verdict: FAIL, 9/10.** Every end-to-end gate passes; the pre-registered bring-up cosine gate misses (min cos 0.99811 vs 0.999).

| gate | v25m | v25 static (SC2 C) | result |
|---|---|---|---|
| TF margin-weighted flips vs v23 CTL1 / vs v25 | W 2.843 (bar 3.078) / W 2.548 (bar 2.895) | W 2.974 vs CTL1 | PASS |
| GSM8K-1319 | 96.82% (1277) | 96.97% (1279) | PASS (−0.15 pt, McNemar p 0.73) |
| BFCL dev / held-out | 556/600 · 1018/1311 | 556/600 · 1018/1311 | PASS (identical) |
| gauntlet · needle-checksum | 12/12 · 500/500 | 12/12 · 500/500 | PASS |
| reasoning_effort codes · image request | identical · reads it | — | PASS |
| per-turn tool emission (bar >= 4 / <= 5) | tool 8, 8 · text 4, 5 · 0 errors | — | PASS |
| bring-up cosine (>= 32 rows) | min **0.99811**, 1 of 1,439 below 0.999 | min 0.99906 | **FAIL** — see cosine-compare.md |
| sidecar timeouts | 0 over 1.03 M waits | 0 | PASS |

Reported, not gated (same instruments as SC2 C): real-text C1 269.2 / 278.9 / 271.6 (v25 280.1 / 283.8 / 273.6, −2%) ·
knee C1 217.1 / 219.8 (v25 201.5 / 218.6) · C8 892.5 / 881.3 (v25 942.5 / 925.1, −5%) · C16 1,429.9 / 1,392.8 (v25 1,404.2 / 1,414.5).
Raw jsonl/json with model outputs are gzipped (`gzip -n`, bytes preserved after path/LAN scrub).
