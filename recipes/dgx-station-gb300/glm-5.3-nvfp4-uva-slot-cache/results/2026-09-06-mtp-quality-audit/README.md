# Public reproducibility package — GB300 MTP quality audit

This directory is a public-safe evidence package for the frozen 2026-09-06 MTP(1) quality audit. It includes inspectable raw receipts, public fixtures, and evaluator scripts sufficient to rerun the local audit summary without access to private filesystem paths or config snapshots.

## Verdict

Overall result: **INCONCLUSIVE / not promoted**. The candidate passed the bounded primary task-loss gate and warm short-C1 speed gate, but failed the frozen secondary gate (`sc13g-mtp` 33/44). MTP remains an experimental performance option, not a quality-approved default.

## Recompute locally

From this directory:

```bash
python3 audit_campaign.py
```

Expected behavior: exit code `1` because the verdict is `INCONCLUSIVE`; stdout should match `campaign-verdict.json` modulo JSON key order/formatting.

The audit script reads only:

- `primary_100_seed20260906.json`
- `results/v1-primary.jsonl`, `results/sc13g-primary.jsonl`, `results/sc13g-mtp-primary.jsonl`
- `results/v1-secondary.jsonl`, `results/sc13g-secondary.jsonl`, `results/sc13g-mtp-secondary.jsonl`
- `results/v1-speed.jsonl`, `results/sc13g-speed.jsonl`, `results/sc13g-mtp-speed.jsonl`

## Published evidence

- Raw primary/secondary/speed JSONL receipts are copied byte-for-byte from the frozen source campaign.
- `audit_campaign.py` is the finite-corpus auditor used for the public recompute path; `public-source/audit_campaign.py` is a duplicate source copy used to recompute the verdict from the JSONL receipts.
- `public-source/secondary_gates.py` is the actual secondary runner, SHA-256 `4c808020713832154347e5747a0db868e00d742ec3fc71d4251aaee714fcc490`, not the older review-baseline copy.
- `public-source/harness.py` is included as the primary scorer/reference harness. Do not run generated HumanEval candidate code on the host; the original harness scores HumanEval only inside Docker sandbox.
- Full config snapshots are excluded. `matched-control-public-proof.json` is a sanitized derivative showing that `sc13g` and `sc13g-mtp` matched after removing only the candidate MTP speculative config.
- `public-evidence-manifest.json` records source and published hashes. Where a file is a derivative, it says so explicitly.

## Privacy handling

No task answers or raw JSONL receipts were silently changed. The public scan in `public-privacy-scan.json` found no private home-path strings, bearer-token values, private-key blocks, or AWS access-key patterns in the published files at packaging time. Config snapshots were not published; only the sanitized matched-control proof was derived from them.
