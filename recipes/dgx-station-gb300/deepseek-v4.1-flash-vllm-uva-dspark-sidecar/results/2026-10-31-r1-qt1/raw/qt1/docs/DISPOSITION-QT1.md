# QT1 disposition — what the orchestrator should do with this

**Do not stamp the temperature>0 gate as complete.** The instruments are built, the controls are run, and
the answer is specific: the sampler is distribution-preserving, but one pre-registered bar fails.

State:
* QT1-A (PIT calibration): **PASS** — KS D = 0.0048, p = 0.284 over 41,705 positions; per-class Holm min
  p = 0.72; no block-level dependence (z = -1.35 / 1.52); NLL within its entropy band.
* QT1-B (decode vs prefill): **PASS** — mean |dlp| 0.0659, top-1 agreement 0.966, Jaccard20 0.808.
* QT1-C (vs h2-REF): **OPEN** — Pg61 repeat floor reproduced (0.0547 / 2.11 % vs 0.0560 / 2.18 %); the
  cross-lane number is blocked by a missing prompt token in the historical receipts, not by the lane.
* P4 (top-1 calibration): **FAIL** — +0.64 pp argmax excess, z = 5.07, entirely inside the low-confidence
  quartile (+2.42 pp, z = 5.27). Equivalent to ~5 % draft-favoured acceptance under the substitution model,
  which is just below the PIT's own MDE (6.9 %). This is the one real defect and it has a named suspect:
  the acceptance path favouring draft proposals where draft and target disagree.

Next actions, in order:
1. Close P7 in the dark window: `qt1_tf.py capture --ref tf-G1-A.jsonl --out tf20-h2REF.jsonl` (15 s on the
   lane), then `qt1_analyze.py --tf-a tf20-Pg61a.jsonl --tf-b tf20-h2REF.jsonl`.
2. Chase P4 in the acceptance path (not the sampler): look for a draft/residue bonus or a stride in
   `rejection_sampler`/`spec_decode` block verification, and re-run `qt1_top1_detail.py` after any change —
   the bar is |z| < 3 and the current reading is 5.07, so the check is sensitive to a real fix.
3. Only then decide on the recipe. If the team accepts a ~5 % draft-favoured argmax excess as immaterial to
   quality, that is a product decision and it must be recorded as an explicit exception, not as a pass.
4. Re-run QT1 on any build that touches the sampler, the spec-decode acceptance rule, or `max_logprobs`
   (raising it above 20 changes the model-config hash, so the tail atom stays coarse by design).

Known limitations to carry into the card text: the gate has low power to a 5 % temperature error (live
control p = 0.033 at 5 %, 0.51 at 1.05), the per-class family has a 5 % FWER by construction (P1 is the
1 % test), the tail atom's internal shape is unmeasured, and nothing here certifies the accept/reject
arithmetic itself (that is the G3 GPU unit test).

Prompts are private Hermes transcripts; nothing in this directory reproduces prompt text, and no results
file may be published.
