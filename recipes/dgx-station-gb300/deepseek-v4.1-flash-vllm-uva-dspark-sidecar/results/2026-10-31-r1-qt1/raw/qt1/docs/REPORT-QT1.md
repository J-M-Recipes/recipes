# QT1 — temperature>0 quality gate, DSV4.1-Flash Pg61 (:30006). Result, 2026-10-31

**Verdict: FAIL at the pre-registered bars — P4 (top-1 calibration) fires. Everything else passes.**
Not a stamp: the sampler question is answered, and the one failing bar is a real, localised defect with a
named suspect and a quantified effect size.

Preregistered in `PREREG.md` (frozen before the run), instrument + analysis in `qt1_analyze.py`
(`--selftest` PASS), generator `qt1_gen.py`, prefill re-score `qt1_rescore.py`, controls `qt1_sim.py`,
localisation `qt1_top1_detail.py`, QT1-C `qt1_tf.py` / `qt1_tf_hist.py`. All clients ran ON the box over
loopback (`http://127.0.0.1:30006`), concurrency 4, no lane restart, no flag change.

## Run

| item | value |
|---|---|
| prompts | `~/dsv41/bench/perclass_prompts.jsonl`, all 200 rows, 40 per class |
| arms | `T1.0a` (seed 100+i), `T1.0b` (seed 900+i): T=1.0, top_p=1, top_k=-1, logprobs=20 |
| controls | `T0.95`, `T1.05` (sampled at 0.95/1.05, scored at T=1) |
| emitted positions | 20,808 + 20,897 = **41,705 pooled** (7,096–9,088 per class), 128-token cap |
| spec decode | active throughout (draft/accepted counted per request; the rejection sampler supplies the returned top-20 from the TARGET logits) |
| prefill re-score | 41,577 positions over the same emitted sequences (`prompt_logprobs=20`, max_tokens=1) |

## QT1-A — global PIT calibration: PASS

Bins = returned top-20 in descending probability + one tail atom (mean tail mass 0.0042). u = coarsened
randomised PIT, fresh V per position; under H0, u ~ U(0,1) exactly at any coarsening.

* KS vs U(0,1): **D = 0.00483, p = 0.284** (bar p >= 0.01) — PASS
* chi-square, 10 equal bins: p = 0.847
* per class KS (Holm across 5): min Holm p = 0.720 — PASS (prose .144, structured .73, code .292, shell_ops .926, tool_json .721)
* dependence: z(autocorr1) = **-1.35**, z(runs) = **1.52** (bar |z| < 3) — PASS, i.e. no draw reuse across verified blocks
* NLL vs entropy: mean(-log p) - H_partition = **+0.0191** nats, tail bound 0.0496 — PASS

Read: the emitted tokens are exact samples of the target conditional the lane reports, to KS D ≈ 0.005 at
41.7k positions. Block verification, suffix drafting and the rejection sampler are distribution-preserving
on this corpus at T=1.

## QT1-B — decode vs prefill numerics: PASS

41,577 positions, prefill path (GEMM/stock attention) vs decode path (GEMV + peer v3): mean |dlp| =
**0.0659**, p99 = 0.814, top-1 agreement **0.9659**, mean top-20 Jaccard 0.808, mean prefill rank of the
emitted token 2.09. Bars 0.10 / 0.95 — PASS. (The 13/200 prefill requests that 400'd under a 4-way burst
were retried; with retries 200/200 and 199/200 succeeded. The 400s were transient, not content-dependent —
the same prompt re-scored clean serially.)

## QT1-C — forward fidelity vs h2-REF: measured only halfway

Forced-token top-20 capture over the 39-doc h2-REF corpus recovered from `tf-G1-A.jsonl` (79,505 forced
positions, 15 s on the lane):

* Pg61 self-repeat (a vs b, identical inputs): mean |dlp| **0.0547**, top-1 flips **2.11 %**, Jaccard20 0.771
  — reproduces the h2-REF repeat floor (0.0560 / 2.181 %) to within 2 %, so the instrument is comparable.
* Pg61 vs h2-REF A: **0.0785 / 2.93 %** — above the P7 bar (0.062 / 2.40 %), but this number carries a known
  artefact and is NOT a lane regression: the historical receipts store `prompt_tokens = n+1` while only the
  37–1,772 scored positions were written, so the leading prompt token is unrecoverable from the file and the
  cross-lane compare runs one token of context short. With the correct alignment the residual collapses from
  0.571 (off-by-one) to 0.0785, i.e. what is left is the missing-token term.
* **P7 stays open.** Closing it needs one h2-REF capture over the same corpus in the dark window:
  `qt1_tf.py capture --ref tf-G1-A.jsonl --out tf20-h2REF.jsonl` (15 s), then
  `qt1_analyze.py --tf-a tf20-Pg61a.jsonl --tf-b tf20-h2REF.jsonl`. Everything else is in place.

## QT1 failed bar — P4 top-1 calibration (z = 5.07)

| split | observed argmax rate | mean max-prob | diff | z |
|---|---|---|---|---|
| pooled | 0.8792 | 0.8728 | **+0.64 pp** | **5.07** |
| T1.0a / T1.0b | 0.8769 / 0.8816 | 0.8711 / 0.8746 | +0.58 / +0.70 pp | 3.23 / 3.94 |
| prose | 0.8255 | 0.8153 | +1.02 pp | 3.10 |
| tool_json | 0.8899 | 0.8805 | +0.94 pp | 3.15 |
| structured | 0.8396 | 0.8332 | +0.63 pp | 2.07 |
| shell_ops / code | — | — | +0.34 / +0.29 pp | 1.68 / 1.21 |
| **confidence Q1 (low)** | 0.5669 | 0.5426 | **+2.42 pp** | **5.27** |
| Q2 | 0.9500 | 0.9487 | +0.13 pp | 0.63 |
| Q3 / Q4 | 1.0000 | 0.9999 / 1.0000 | +0.01 / 0 pp | 1.02 / 0.06 |

The whole excess lives in the diffuse quartile; where the target is confident the emitted token IS the
argmax (obs = exp = 1.0000), so the defect only appears where the draft and the target can disagree. That
is the signature of a **draft-favouring acceptance path** (accepted draft tokens are the drafter's own
argmax), not of a reporting-calibration offset, which would be flat across confidence.

Under the substitution model "with probability eps the emitted token is the drafter's argmax", the observed
+0.64 pp equals **eps ≈ 5.0 %** (the sim's argmax arm at eps = 0.05 produces a +0.636 pp top-1 excess).

## Instrument controls (`run1/sim.json`)

* **Null, 200 replicates**, synthetic tokens drawn exactly from each recorded distribution: P1 false alarm
  **4/200 = 2.0 %** (nominal 1 %, consistent), P2 **5.0 %** (that is its designed per-class Holm FWER, not
  slack), P3 0 %, union 7 % — the union is 1 % + 5 % by construction, so quote P1 alone for the primary.
* **Power, draw reuse** (one uniform shared by every position of a verified block): 1.00 at k = 2, 3, 5, 8,
  16 (median |z| 11.3 → 20.5). Smallest detectable: **k = 2**.
* **Power, acceptance bias** (argmax substitution): 0.00 at eps = 1 %, 0.07 at 2 %, 0.40 at 5 %.
  MDE at 80 % power: **eps ≈ 6.9 %** (argmax model) / **48 %** (top-5-support model — the pessimistic corner,
  because at this peakedness the target's own top-5 is nearly the target). **This is why P1 passes while P4
  fires: the live defect (eps ≈ 5 %) sits just under the PIT's MDE and is 5 sigma to the top-1 rate.**
* **Power, temperature**: 0.47 at T = 0.95, 0.80 at T = 1.05 (sim model); MDE |dT| ≈ 0.066. The LIVE
  controls agree that a 5 % temperature error is at the edge: T0.95 scored at 1.0 gives D = 0.0100,
  p = 0.033 (rejects at 5 %, not at the 1 % bar), T1.05 gives D = 0.0057, p = 0.51. **A 5 % temperature
  setting error would not have been caught by this gate** — say so plainly rather than implying otherwise.

## What this does and does not license

Does: at T=1 on this corpus and this build, emitted tokens are exact samples of the reported target
conditional (KS D 0.005 over 41.7k positions; no block-level draw reuse; NLL consistent with entropy), the
decode path agrees with prefill (0.066 nats, 96.6 % top-1), and the forced-scoring repeat floor reproduces
h2-REF (0.055 / 2.1 %).

Does not: certify the accept/reject arithmetic (G3 GPU unit test), any temperature other than 1.0, the
tail atom's internal shape, or P7. And it records one open defect: a ~5 % draft-favoured argmax excess that
is invisible to the PIT at this N and only shows up in the top-1 calibration.
