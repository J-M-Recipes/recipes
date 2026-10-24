# SC3/SC3b result — cold-expert placement — 2026-10-21 22:00 CDT

| Gate boot (cosine, real text, frozen 38-prompt set sha1 7eaab6b7) | checks ≥32 rows | full-prefill min cos | T=128 min | T=128 p01 | T=128 median | below 0.999 | KV tokens | Verdict |
|---|---|---|---|---|---|---|---|---|
| SC0 arm A — ours 295/89 (static-v1) | 1,341 | 0.99921 | 0.99906 | 0.99963 | 0.99983 | 0 | 2,536,059 | PASS |
| SC3 GMIX — el8 mix-v1 285/99 (positive control) | 1,179 | 0.99958 | 0.99911 | 0.99936 | 0.99980 | 0 | ~6.09M (SC0 arm B boot) | PASS |
| SC3b G285 — ours nested 285/99 (static-v1-h285) | 1,251 | 0.99947 | **0.99885** | 0.99963 | 0.99983 | **2** (L39, T=128, 45 and 42 rows: 0.99885, 0.99893) | **6,094,432** | **FAIL (pre-registered bar)** |

(SC3's first G285 attempt at 21:24 was killed by a sibling session's stop_all — not a result.)

Reading, labelled as such: the G285 distribution is indistinguishable from arm A's at p01/median and better on full prefill; the two
failing checks are layer-39 replay-trimmed (T=128) checks just above the 32-row cutoff — the same class that set arm A's 0.99906 and
GMIX's 0.99911. The 0.999 bar sits inside the tail noise of this check class. **The bar was pre-registered, so 285/99-ours is NOT
adopted.** Re-gating it under a noise-calibrated bar (e.g. a same-map repeat boot to measure the tail, or per-layer p01) would be a
new registration and is James's call. P arms were skipped by design → no 285/99 speed data from SC3b.

Decision (pre-registered fallback): SC2 runs on **295/89 (rowmap-static-v1.json)**, KV 2.54M (clears the 2.50M bar).
Open option for James: el8 mix-v1 285/99 passed its gate and gives ~6.09M KV, but its speed with the sidecar is unmeasured (it was
the slower map Grace-only: M3G-mix C1 147 vs ours 176) — would need its own P-arm bracket.
