# E1 v2 live profile

This directory publishes the September 8, 2026 GB300 E1 v2 live profile package for GLM-5.3 NVFP4 UVA slot-cache.

- `OUTCOME.md` is the narrative result and limitations summary.
- `analysis/` contains the offline corrected-bucketing/API/masked-copy analysis outputs copied from `e1v2-analysis/`.
- `live-receipts/` contains the public live receipt subset copied from `e1v2-receipts/`.
- `RECEIPTS-EXCLUDED.md` lists local files intentionally excluded from git publication and Station-only raw artifacts.
- `analysis-SHA256SUMS` and `live-receipts-SHA256SUMS` hash the copied publication payloads.

Contract verdict: **INCONCLUSIVE by design**. The live collection succeeded; decode-only attribution was corrected after separating prefill bypass kernels: decode is about `40.1 ms/step` wall and `43.1 ms/step` aggregate GPU, with `masked_row_copy` at `20.10 ms/step` (46.6% of decode GPU) and routed MoE decode GEMMs at `5.38 ms/step`; the bookkeeping hypothesis remains falsified (`fused_bookkeeping + scalar_gather = 1.557518722 ms/step`).
