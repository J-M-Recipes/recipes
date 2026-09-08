# E1 v2 live profile

This directory publishes the September 8, 2026 GB300 E1 v2 live profile package for GLM-5.3 NVFP4 UVA slot-cache.

- `OUTCOME.md` is the narrative result and limitations summary.
- `analysis/` contains the offline corrected-bucketing/API/masked-copy analysis outputs copied from `e1v2-analysis/`.
- `live-receipts/` contains the public live receipt subset copied from `e1v2-receipts/`.
- `RECEIPTS-EXCLUDED.md` lists local files intentionally excluded from git publication and Station-only raw artifacts.
- `analysis-SHA256SUMS` and `live-receipts-SHA256SUMS` hash the copied publication payloads.

Contract verdict: **INCONCLUSIVE by design**. The live collection succeeded; the offline analysis falsified the bookkeeping hypothesis (`fused_bookkeeping + scalar_gather = 1.557518722 ms/step`) and identified `routed_moe`, `masked_row_copy`, and `dense_gemm` as the dominant corrected GPU buckets.
