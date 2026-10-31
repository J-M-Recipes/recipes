# QT1-CONFIRM VERDICT: PASS (2026-10-31 12:49 CDT)

Pre-registration: `PREREG-CONFIRM.md` (sha256 8ae41fd2…). Frozen code hashes verified before scoring (`shasum -c`
OK for qt1_analyze.py, qt1_exact.py, qt1_gen.py). Lane: Pg61 on :30006 after Cards R1/R1b. Soak clock started at
12:44:08 CDT, so this run is inside the soak. Box loopback, C4. The p3 watchdog reads `trip: null` at the end.

Arms, run 12:45–12:48 CDT, 200 records each, 0 errors, max_tokens 256 (EOS stops early on many rows):
| arm | T | seed base | positions |
|---|---|---|---|
| C1 | 1.0 | 310000 | 29,848 |
| C2 | 1.0 | 320000 | 30,225 |
| K95 | 0.95 | 330000 | 29,535 |
| K105 | 1.05 | 340000 | 30,269 |

| id | test | result | bar | |
|---|---|---|---|---|
| Q1 | M mid-PIT mean z | −0.211 | \|z\| < 3 | PASS |
| Q2 | R rank-calibration chi² p | 0.7355 | ≥ 0.01 | PASS |
| Q3 | R per class, Holm min p | 0.7446 | ≥ 0.05 | PASS |
| Q4 | tie-fair top-1 z | 0.193 (obs 0.8614, exp 0.8612) | \|z\| < 3 | PASS |
| Q5 | dependence z (autocorr / runs) | −0.009 / −0.171 | both \|z\| < 3 | PASS |
| Q6 | NLL − H_partition | +0.0199 nats | [−0.02, 0.0714] | PASS |
| S1 | control K95 scored at T=1, M z | −5.31 (R p 2.6e-4) | ≤ −3 | DETECTED |
| S2 | control K105 scored at T=1, M z | +5.33 (R p 3.6e-6) | ≥ +3 | DETECTED |

Supporting, not gating: randomised-PIT KS pooled p 0.210; per-class KS Holm min 0.481.

**Verdict: PASS.** Allowed claim (from the PREREG): sampler output at T=1.0, top_p=1 is calibrated to the reported
target distribution on the speculative-decode path, within the detectable effect. Argmax substitution below ~2 %/token
and |ΔT| below ~5 % are not excluded. Not covered: other temperatures, top_p/top_k truncation, the accept/reject
arithmetic in isolation (Card G3), anything outside the reported top-20.

Discovery run (run1): registered P4 FAIL, kept on file with its diagnosis in ORCH-REVIEW.md.
