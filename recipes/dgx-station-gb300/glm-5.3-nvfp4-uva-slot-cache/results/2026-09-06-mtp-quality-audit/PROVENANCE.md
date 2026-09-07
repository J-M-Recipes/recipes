# Public provenance — GB300 MTP quality audit

Source campaign public label: `<source-campaign>`.
Actual secondary runner public label: `<secondary-runner-source>/secondary_gates.py`.

This bundle is **not** a byte-identical copy of the private working tree. It is a public evidence package:

- Raw JSONL receipts and public fixtures/evaluator sources are byte-identical copies; hashes are in `public-evidence-manifest.json`.
- Markdown documentation and matched-control config proof are public derivatives; they intentionally avoid private home paths and full config snapshots.
- Full config snapshots are excluded from publication. `matched-control-public-proof.json` retains only non-secret image/command equality evidence and explicitly lists excluded fields.

Actual secondary runner hash:

- `public-source/secondary_gates.py`: `4c808020713832154347e5747a0db868e00d742ec3fc71d4251aaee714fcc490`

Conclusion remains unchanged: primary and speed gates pass; overall promotion remains **INCONCLUSIVE** because secondary fails.
