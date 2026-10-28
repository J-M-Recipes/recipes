# Card H1 — L2 attention/indexer key-sharing + #61161 backport vs v27 (2026-10-27, 13:11–14:24 CDT)

**RESULT: SUBBAR** — real, bit-identical, and 9 µs short of the pre-registered 0.10 ms bar.
Receipts: box `~/dsv41/m3/h1/results-cardH1/` (`verdict.json`, `VERDICT.txt`, per-arm dirs). Arms A → C → A2,
bracketed, lane restored 14:23:43 (KV 4,400,013, args/env identical to v27), watchdog pid 478262 PPID 1, lane check
rc=0, model bound.

## Gates (pre-registered, frozen before boot A)

| gate | result | numbers |
|---|---|---|
| G0 validity | **PASS** | A/C/A2 all valid: KV 4,400,013, args == v27 on every arm, env == v27 + the five MEGA_L2 vars (0 on controls, C's values on C), 0 foreign POSTs, 0 errors, 0 sidecar timeouts |
| L2 engagement (C) | **PASS** | `topk_src=40, topk_hit=150, topk_miss=0, pos_hit=312, pos_miss=0, verify_ok=150, verify_bad=0`; 0 `VERIFY MISMATCH` lines — every shared table was re-checked with `torch.equal` against a fresh computation and matched |
| P1 (#61161 GPU tests) | **PASS** | overlay run 61 passed; stock image negative control 2 failed / 2 passed as expected |
| G1 controls | **PASS** | A 11.208, A2 11.157 ms, spread **0.46%** (bar 2%) |
| G4 step (BRRR 2K C1, T=0, 5 reps) | **FAIL** | **C 11.0655 ms vs min(ctrl) 11.1566 → −0.0911 ms** (bar ≤ −0.100); vs control mean 11.1824 → **−0.1169 ms** |
| G3 per-class (5 classes × C1/C8) | **PASS** | ratios 0.980–1.034× of control mean; worst cell structured_c1 0.980× vs a 2.05% tolerance |
| G1c fidelity (TF) | **PASS** | flips C-vs-A 2.176%, C-vs-A2 2.067%, floor 2.121% (bar floor+0.30 pt); margin gate PASS (0 confident agent/tool flips, W 2.842 vs calib 2.674) |
| G2 gauntlet | **PASS** | C 12/12 (A 12/12, A2 12/12) |

Per-rep steps, C: 11.079, 11.070, 11.062, 11.066, 11.048 — a 0.031 ms spread, tighter than either control (A 0.027,
A2 0.034). The effect is 4.5× the control half-spread, so it is not a lucky run; the *bar* was the casualty, not the
claim.

Report-only: BRRR 2K C1 **T=1.0** C 11.058 ms / 463.2 tok/s / 5.12 tok/step vs A 11.244 / 457.4 / 5.15 and A2 11.191 /
463.1 / 5.18. Sidecar stall identical on all arms (1.20–1.22 ms/step), as expected — nothing here touches the sidecar.

## What it means

- **L2 is real and free.** −0.117 ms/step on the control mean = **+1.05%** C1 at our 5.24 tok/step, achieved with
  bit-identical output (150/150 shared tables verified equal, greedy/TF at the control floor, gauntlet clean). The
  card's own prediction was −0.121 ms; measured −0.117.
- **Why the bar missed:** the pre-registered bar was `min(A,A2) − 0.10`, and A2 happened to land 51 µs faster than A.
  Against the control *mean* the effect clears the bar with room. The card anticipated this exact shape ("a G4 miss by
  < 0.03 ms with G1–G3 passing is 'L2 real but sub-bar'. It is not a regression").
- **#61161 is carried for exactness, not speed.** It removes the reused-draw bias in block verification at temp > 0;
  at T=0 it is inert by construction, which is why G4 was measured at T=0. 61/61 of its suite passes on our pin and the
  stock image fails the two block-verification params — the negative control that proves the test has teeth.
- **Nothing regressed.** Ten per-class cells within 2%, all three arms 12/12 on the gauntlet, fidelity at the
  control floor.

## Recommendation (James's call, never the card's)

Promote v27 + `hook-h1` + `overlay-61161` as the new reference: it is ~1% faster on the class we are trying to catch
and it removes a known sampling bias at temp > 0, with no measured cost anywhere. If he prefers the letter of the
pre-registered bar, the alternative is to keep v27 and stack L2 with the next lever — its cost is ~0, so it can ride
along either way.

Next in the revised order (see `REPORT.md`): L1 dense-GEMM microbench (needs its 8.5 min GPU window — predict
≥0.08 ms/step or skip), then the RTX-sidecar small-M kernel trace (the 80 µs/call finding from `P2`), then k=7.
