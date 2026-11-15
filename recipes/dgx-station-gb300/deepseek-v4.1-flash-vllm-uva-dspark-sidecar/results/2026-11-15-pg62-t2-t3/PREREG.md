# PREREG — T2/T3 quality suite on the live Pg62 lane (written 2026-11-15 before the first request)

Lane: Pg62 on :30006 (serving since 2026-11-14 23:11:55 CST, in its 48 h soak). Client: the M4 over the LAN,
concurrency <= 4, read-only, nothing started or stopped on the box. Tools: llm-inference-bench (`~/hermes/lil-bench`,
version recorded in bench-version.txt), tool-eval-bench at the bench's pin, `bfcl_gate.py` (sha1 recorded in facts.txt).

Reference = Pg61's own runs, paired by item id. They are from 2026-10-29 (Cards G3/G4), not this window: this is a
cross-day comparison against stored per-item results, not a same-window control. Pg61 cannot be re-run without ending
the soak. Pg62 differs from Pg61 in decode-attention bytes on some calls, so near-tie items are expected to move both ways.

Order: T0 -> needle-checksum 500 (C4) -> GSM8K full 1,319 (C4) -> tool-eval hardmode x1 (parallel 4) ->
BFCL held-out (live_simple 258 + live_multiple 1,053; CONC 4, MAXTOK 2048, THINKING 0) -> MMLU-Pro 1000 (C4).

Bars (a FAIL is published as a FAIL; the lane decision is James's):
1. T0: model id listed, exact-word chat, one parsed tool call.
2. needle-checksum: EXACT 500/500.
3. GSM8K-1319 vs Pg61 Card G4 (1279/1319), paired: accuracy delta >= -1.0 pt (recipe bar gsm8k_delta_pt_min) and no
   exact two-sided McNemar loss at p < 0.05.
4. BFCL held-out vs Pg61 Card G3 (1026/1311), paired: delta >= -1.0 pt (recipe bar bfcl_delta_pt_min); McNemar reported.
5. MMLU-Pro-1000 vs Pg61 Card G4 (87.7 %), paired: no exact two-sided McNemar loss at p < 0.05.
6. p3 watchdog: trip null, v2/v3 timeout counters 0, both containers up after the suite.
Reported, no bar: tool-eval score and failing TC ids vs Pg61's 89 / 91; engine load during the window (/metrics every 5 s).
Not exercised: temperature > 0 quality, teacher-forced bracket, gauntlet, contexts > the needle profile's.
