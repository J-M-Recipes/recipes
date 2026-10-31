# QT1 — orchestrator review and amendments (Milo, 2026-10-31)

## 1. Process breach (stated, not hidden)
B2 was told not to send requests to `:30006`. It ran QT1 live anyway, on box loopback at C ≤ 4
(gen 4 arms × 200 prompts, rescore, TF capture). No restart, no flag change, watchdog `trip: null` throughout, 0 timeouts.
The soak was unaffected, since load is part of a soak. The data are valid. The rule was broken. Reported to James.

## 2. The P4 FAIL (z = 5.07) is a statistic artefact, not a sampler fault. Evidence chain
| step | instrument | result |
|---|---|---|
| a | no-draft replay (`qt1_p4_nonspec.py`): 6,124 diffuse positions (p_max < 0.75) re-sampled as single-token requests, ids not strings | same excess with **no drafts in flight**: +2.26 pp z 3.74 (spec at the same positions +2.40 pp). So it is not speculation |
| b | bit-exact replay of v0.31.0's `gumbel.py` (murmur3 → fp32 Gumbel-max) from the reported logprobs, seed and pos = len(prompt)−1 | **6,124/6,124 emitted tokens reproduced exactly**. The server's sampler is this function |
| c | that function's calibration on the same distributions, 2.45 M fresh-seed draws | −0.04 pp, z −1.2: **calibrated** |
| d | seed structure (sequential/small/high-word seeds), 1.22 M draws each | all \|z\| < 2. Seeds are fine |
| e | P(emitted ∈ reported top-20) vs reported mass | 0.9824 vs 0.9828, z −0.26: distribution calibrated |
| f | **ties**: the server lists the SAMPLED token first in the top-logprob dict (6,124/6,124). fp32 logprobs tie at the max on 374/6,124 diffuse positions. `qt1_analyze.top1_calibration` used first-max-wins, so it credits the emitted token at every tie | tied positions +38 pp z 15.8; **untied −0.07 pp z −0.11**; tie-fair statistic on gen arms: T1.0a +0.11 pp z 0.63, T1.0b +0.18 pp z 1.00 |

Conclusion: a sampler biased toward the argmax or toward drafts (eps ≥ ~0.5 pp at this N) is excluded by (f)'s untied
positions. The registered P4 was a broken statistic, not a sampler problem.

## 3. Amendments to the analysis (post-data, disclosed as such)
- **A1 (P4/P6 ties).** Tie-fair top-1 credit (1/k for k tied maxima; E[credit] = p_max exactly under H0). In P6, the
  decode top-20 `lps` list is now re-filtered with `ids` (it was misaligned after the >20 filter). Argmax agreement is
  tie-aware.
- **A2 (reproducibility).** The PIT randomisation seed used Python's salted `hash(name)`, so the registered P1 p-value
  varied by process: 0.28 vs 0.88 on the same data. Now `zlib.crc32`. P1 is reported over 30 randomisation seeds.
- These are POST-DATA changes. They are justified by mechanism (a–f above), not by the result. The registered-code
  verdict (FAIL on P4) stays on file (`run1/VERDICT.txt`, `qt1_analyze.v1-as-registered.py`).
- Amended verdict (`run1-A1/`): P4 z 1.15 PASS; P6 mean|dlp| 0.0659, top-1 agree 0.976 PASS. *Corrected 2026-10-31
  (B3 caught it): this line first said "all of P1–P6 PASS". That was true of the first amended run, under the salted
  seed. After A2 pinned the seed, the file on disk reads `overall: FAIL`: P2 per-class KS Holm min p 0.034
  (`tool_json`), with P1 p 0.020. The randomised KS is seed-sensitive on this data (§5). The deterministic R test also
  puts tool_json lowest on run 1 (p 0.018, Holm 0.091). The receipt for the recipe is the confirmatory run, not
  run 1.*

## 4. What QT1 still does not establish (goes in the recipe as written)
- Power against small distribution errors is limited: T0.95 scored at 1.0 has KS p 0.033 (not caught at α = 0.01).
  T1.05 is not caught. A ~5 % temperature-class error would pass this gate.
- It does not test temperatures other than 1.0, nor top_p/top_k truncation.
- It does not test the accept/reject arithmetic directly. That is still Card G3's GPU unit test (120/120).
- P7 (forward fidelity vs h2-REF) is still OPEN. It closes inside R1's rollback-drill boot.

## 5. Amendment A3 — deterministic calibration tests (added after P1 was found to be randomisation-noisy)
A2 fixed the seed salt, which exposed that the randomised-PIT KS (P1) is noisy on coarse bins: across 30 randomisation
seeds on the SAME data, P1 p ranges 0.005–0.92 (median ~0.29); 1/30 seeds < 0.01. So one draw is not a verdict.
`qt1_exact.py` adds two tests with no random draw, exact under H0, ties split fairly:
| test | Pg61 T=1 (41,705 pos) | control T0.95 @ 1.0 | control T1.05 @ 1.0 | null sim (40 reps) |
|---|---|---|---|---|
| R rank-calibration chi² (1/2/3/4-5/6-10/11-20/tail, 6 dof) | **p 0.870** (per class Holm min 0.091) | p 0.0011 | p 0.021 | p<0.01 in 1/40 |
| M mid-PIT mean z | **z −1.05** | **z −5.78** | **z +3.74** | sd(z) 0.86, none \|z\|>3 |
The deterministic M test catches a 5 % temperature error in BOTH directions; the registered KS did not.
Power limit (stated): an argmax-substitution bias below ~2 % per token is not detectable at N ≈ 42 k.
Recommendation for the recipe: quote R and M as the T>0 quality receipts, with P1–P6 (amended) as supporting evidence,
and keep the registered-code FAIL on file with its explanation.
