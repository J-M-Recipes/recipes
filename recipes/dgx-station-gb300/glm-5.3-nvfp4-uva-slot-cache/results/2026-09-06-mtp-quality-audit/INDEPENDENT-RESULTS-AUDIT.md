# Independent results audit — public-safe summary

## Verdict

Overall promotion remains **blocked / INCONCLUSIVE** under `CONTRACT.md` because the secondary gate does not pass in any lane, including the candidate (`sc13g-mtp`: 33/44).

- Primary code/math gate: `sc13g-mtp` versus matched `sc13g` has 1 loss, 1 win, 98 ties over 100 unique tasks; one-sided 95% Clopper-Pearson gross-loss upper bound `0.04655981145353899` (< 0.05). Both lanes scored 190/200 repeats.
- Speed gate: warm short-C1 median effective TPS `53.98028929327154` for `sc13g-mtp` vs `40.92076273384625` for `sc13g`, gain `0.31914181669501307`; long median wall ratio `0.7696895272711062`.
- Secondary gate: `v1` 36/44, `sc13g` 35/44, `sc13g-mtp` 33/44. All lanes fail the all-pass secondary promotion rule.

## Public recomputation

Run `python3 audit_campaign.py` from this directory. The expected exit code is `1` for `INCONCLUSIVE`; the expected JSON is `campaign-verdict.json`.

## Evidence now included

This public bundle includes byte-identical raw primary, secondary, and speed JSONL receipts for all three lanes, the primary fixture set, the actual secondary runner, and the finite-corpus auditor. The older private-path-only evidence references have been replaced by inspectable files and hashes in `public-evidence-manifest.json`.

## Limits

This supports only a bounded observation: MTP(1) improved this measured short-prompt speed lane and did not show primary code/math degradation under the frozen low-reasoning, temperature-0 task-loss rule. It does not support an overall promotion, broad quality claim, token-equivalence proof, max-context qualification, or C4/C8 quality claim.
