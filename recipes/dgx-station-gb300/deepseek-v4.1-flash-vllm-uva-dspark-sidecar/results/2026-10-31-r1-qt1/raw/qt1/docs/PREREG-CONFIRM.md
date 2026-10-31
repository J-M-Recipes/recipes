# QT1-CONFIRM — confirmatory T>0 sampler-calibration run (PRE-REGISTERED 2026-10-31 ~12:20 CDT, before any data)

**Why.** The discovery run (run1) FAILED its registered P4. The cause was traced to a tie artefact in how top-logprobs
are reported (ORCH-REVIEW.md). The fixes (tie-fair credit, deterministic seed, the deterministic R/M tests) were
chosen after seeing run1. Advisor consult-3: those amended tests can only become the recipe's receipt after a fresh
run that passes them with every rule fixed in advance. This is that run.

**Lane.** Pg61 on :30006, run after Cards R1/R1b restore it. Box loopback, C = 4, no other load from us. Every request
uses `/v1/completions`, T as below, top_p = 1, top_k = −1, logprobs = 20, max_tokens = **256** (run1 used 128).

**Arms.** Fresh seeds. Same 200 per-class prompts: the pool is fixed at 200, so the prompts are reused and NOT new.
| arm | T sampled | scored at | seed base |
|---|---|---|---|
| C1 | 1.0 | 1.0 | 310000 |
| C2 | 1.0 | 1.0 | 320000 |
| K95 (control) | 0.95 | 1.0 | 330000 |
| K105 (control) | 1.05 | 1.0 | 340000 |

**Analysis code, frozen now.** `qt1_analyze.py` at sha256 below (amendments A1+A2), `qt1_exact.py` at sha256 below.
No change after data. A crash is fixed only if the fix cannot change a number (stated in the report).

**Primary bars (C1 ∪ C2 pooled). Each must pass.**
| id | test | bar |
|---|---|---|
| Q1 | M: mid-PIT mean z (ties split) | \|z\| < 3 |
| Q2 | R: rank-calibration chi², 7 categories (1,2,3,4-5,6-10,11-20,tail) | p ≥ 0.01 |
| Q3 | R per class (5), Holm | min Holm p ≥ 0.05 |
| Q4 | tie-fair top-1 rate vs mean p_max (P4 amended) | \|z\| < 3 |
| Q5 | dependence (P3): lag-1 autocorr z and runs z of the PIT | both \|z\| < 3 |
| Q6 | NLL − H_partition (P5) | in [−0.02, tail bound + 0.02] |

**Sensitivity bars (controls). Each must be DETECTED, otherwise the run is INCONCLUSIVE, not PASS.**
| id | rule |
|---|---|
| S1 | K95 scored at T=1: M z ≤ −3 |
| S2 | K105 scored at T=1: M z ≥ +3 |

**Supporting, reported but not gating.** The randomised-PIT KS (P1) as the distribution of p over 30 randomisation
seeds (20261032..20261061). QT1-B decode-vs-prefill on C1 if the rescore runs (P6 bars from PREREG.md).

**Verdict.** PASS = Q1–Q6 pass AND S1, S2 detected. FAIL = any Q fails. INCONCLUSIVE = all Q pass but a control is
not detected.

**Claim allowed on PASS.** "Sampler output at T=1.0, top_p=1 is calibrated to the reported target distribution on the
speculative-decode path, within the detectable effect: argmax substitution below ~2 %/token and \|ΔT\| below ~5 % are
not excluded."
**Not covered:** other temperatures, top_p/top_k truncation, the accept/reject arithmetic in isolation (Card G3), and
anything outside the reported top-20.

## Frozen code
```
86c150980067a4b3db9096ed2d3c939b5a3db5be4c983c631988e78a16ee9c1b  qt1_analyze.py
6293774a1556b01806fe7517dfbcf9e39acd59b8126dcc38189c603e48197d44  qt1_exact.py
3d6b9e1dab22a1bf7614980910f7ebce69d64b6d8caf678ed7c0278884756509  qt1_gen.py
```
