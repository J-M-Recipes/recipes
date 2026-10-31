# QT1 P7 — forward fidelity, Pg61 vs h2-REF (PRE-REGISTERED 2026-10-31, before h2-REF is booted)

**Question.** On the same fixed token corpus (`tf-G1-A.jsonl`: 39 docs, 79,505 forced positions), does Pg61 put out
the same next-token distributions as the incumbent h2-REF, within the two lanes' own repeat noise?

**Captures.** All via `qt1_tf.py capture` (prompt_logprobs = 20, T = 0, C = 1, box loopback).
- Pg61 a and b: already taken on the live lane 2026-10-31.
- h2-REF a and b: taken inside Card R1 phase P4, on h2-REF as booted by the rollback script, no other load.

**Statistics.** `qt1_analyze.tf_compare` with amendment A1 (tie-aware top-1). Computed BEFORE h2-REF exists.
- Pg61 floor (a vs b): mean |Δlp| 0.05468, tie-aware top-1 flip rate 1.532 %, Jaccard@20 0.771.

**Bars (a FAIL on any one fails P7).**
| bar | rule |
|---|---|
| P7a | mean \|Δlp\| (Pg61a vs h2a) ≤ 1.25 × max(Pg61 floor, h2 floor) |
| P7b | top-1 flip rate (Pg61a vs h2a) ≤ max(floor flip rates) + 0.50 pp |
| P7c | h2 floor itself ≤ 0.08 mean \|Δlp\| (else the h2-REF capture is broken; P7 = INVALID, not PASS) |
| P7d | ≥ 79,000 aligned positions in every compare (an alignment defect shows up as ~99 % flips; that run is INVALID) |

**What this can and cannot conclude.**
- It CAN show forward-pass numerics on the prefill path are within run-to-run noise of the incumbent.
- It CANNOT speak to decode graphs or speculation. Those are QT1-A/B on Pg61, already PASS after amendment.
- The historical cross-lane number (0.0785 / 2.93 %) is withdrawn. B2 found a one-token alignment defect in the old
  receipts. The 99.5 % "flip" rates in `run1/tf-Pg61-vs-*.json` come from that defect and are INVALID.
