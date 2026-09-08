# E1 v2 live profile — 2026-09-08

Final state: **profile collected; contract verdict INCONCLUSIVE by design; decode-only attribution corrected after separating prefill bypass kernels; offline analysis falsified the bookkeeping hypothesis.**

## Timeline

| time (CDT) | event |
|---|---|
| 2026-09-08 11:10:53 | E1 v2 live window started on GB300, run id `e0-e1-20260908-v2`. |
| 11:11 | Incumbent stopped. |
| 11:33 | Candidate `glm53-big-e1-nsys` ready after about 22 minutes cold load; image ID `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`. |
| 11:33–11:45 | Acceptance probe and C1/C4/C8 benches ran. |
| 11:45:27–11:45:35 | Nsight START/STOP ACK window captured. |
| 11:45–12:06 | Incumbent restore ran. |
| 12:06 | Restore proof returned `WINDOW_RESTORE_OK`; proof id `c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea`. |
| 12:07 | Runner exited `WINDOW_E1_V2_COLLECTION_OK`; release closed, failsafe stopped, credential scan reported 0 hits. |

Scope: K=1 only. The offline analysis files in `analysis/` are derived from the live receipts; `live-receipts/nsys-buckets.json` is intentionally preserved as the live runner produced it with the pre-fix regex.

## Contract verdict

The frozen E1 v2 contract verdict is **INCONCLUSIVE**, not failed:

- the v2 live runner was collection-only;
- it intentionally fails closed without engine-core/graph-node evidence;
- it intentionally fails closed without hash-bound CUDA API attribution in the live receipt directory;
- the corrected bucketing and CUDA API attribution are offline analysis, not retroactive live-contract evidence.

The live `e1-verdict.json` issues are:

```text
missing engine-core/graph-node evidence
missing hash-bound CUDA API attribution
```

## Comparator vs v1 E0 incumbent

| metric | v1 E0 incumbent | E1 v2 candidate | delta / note |
|---|---:|---:|---|
| Weighted K=1 accepted length | 1.826940 | 1.820444 | -0.006496; both stop-gate false |
| Prose C1 agg tok/s | 45.75–46.33 (mean 45.99) | 45.62–47.11 (mean 46.143) | similar |
| Prose C4 agg tok/s | 45.12–45.51 (mean 45.367) | 45.44–46.32 (mean 45.877) | similar |
| Prose C8 agg tok/s | 45.29–45.54 (mean 45.41) | 45.55–46.05 (mean 45.883) | similar |
| Code C1/C4/C8 agg tok/s | 42.95 / 42.95 / 42.84 | 42.89 / 43.12 / 43.14 | similar |
| Profiler slowdown | — | 1.5846323% in live verdict; 1.61% from offline summary | acceptable for profiling |

## Falsified hypothesis

The targeted bookkeeping hypothesis is falsified: `fused_bookkeeping + scalar_gather = 1.557518722 ms/step`, below the 2.0 ms/step opportunity gate.

Correction, September 8, 2026: the first cut divided the whole capture by the 140 verification steps and misattributed prompt/prefill bypass kernels to decode. Separating rows by instance count gives decode wall `256 tokens / 45.648 tok/s / 140 = 40.06 ms/step` and decode aggregate GPU `(8389.2 - 2351.4) / 140 = 43.13 ms/step`. The decode path is still GPU-bound because aggregate GPU work exceeds decode wall; kernel summaries aggregate overlapped GPU work and are not a critical-path decomposition.

## Correction 2026-09-08: decode-only attribution

The first published table was a corrected whole-capture GPU bucket table, not a decode-only attribution table. It included four prompt/prefill rows: two `Instances == 300` bypass MoE GEMM rows (`4 requests * 75 cached layers`) plus two `Instances == 4` prompt rows. See `analysis/decode-only-attribution.md` and reproduce with `analysis/decode_only_buckets.py`.

Second correction: the first cut also misread vLLM `backend=eager` as eager decode execution. It only disables Inductor. Decode is CUDA-graph captured: the decode rows show about `4407` kernel instances/step, while the API trace shows about `189` kernel-launch API calls/step plus `3.06` `cudaGraphLaunch_v10000` calls/step. Therefore roughly `4200` kernels/step are replayed inside CUDA graphs. The `5.52 ms/step` launch-API time is mostly overlapped and is not a first-order lever.

| bucket | ms/step | % decode GPU | instances | rows |
|---|---:|---:|---:|---:|
| masked_row_copy | 20.10 | 46.6% | 42,900 | 1 |
| dense_gemm | 9.41 | 21.8% | 124,007 | 27 |
| other | 7.18 | 16.6% | 417,934 | 73 |
| routed_moe | 5.38 | 12.5% | 21,450 | 2 |
| fused_bookkeeping | 1.05 | 2.4% | 10,725 | 1 |

The prefill-attributed aggregate GPU total is 2351.4 ms. The full profiled wall was 8.218 s; decode reconstructed from throughput is 5.608 s; the residual `8.218 - 5.608 = 2.61 s` is prefill plus TTFT for four prompts, about 0.65 s each. That matches the bypass cost of about `5.2 + 2.6 = 7.8 ms/layer` across 75 cached layers, or about 0.59 s per prompt before smaller overheads.

## CUDA API attribution

From `analysis/api-attribution.json`:

- total CUDA API time: `36.670742 ms/step`;
- `cudaEventSynchronize`: `29.867015 ms/step` over `3.128571` calls/step;
- kernel launch API time, excluding graph launches: `5.515798 ms/step` over `189.014286` launch APIs/step;
- `cudaGraphLaunch_v10000`: `3.064286` calls/step and `0.361041 ms/step`;
- `cudaLaunchKernel`: `123.778571` calls/step and `3.150933 ms/step`.

Caveat: `cudaEventSynchronize` duration is host blocking/wait time. It is not proof of separate CPU work that can be added to the GPU kernel critical path.

## `masked_row_copy` characterization

`masked_row_copy` is a dominant bucket at `20.097824 ms/step` in the kernel summary. The full GPU trace has `42,900 = 143 * 75 * 4` calls. Treating the last 140 inferred slot-cache steps as the verification window gives `42,000 / 140 = 300` calls/step: 75 cached layers times four launches.

Per-call distribution over all rows: min `0.992 us`, p50 `34.144 us`, p95 `214.496 us`, p99 `323.616 us`, max `519.168 us`. The earlier about-15% near-empty note was dominated by the small scale-tensor launches; only about `2.4%` of w13 launches are below 10 us. The launch pattern is fixed `GrdY=(768, 384, 96, 48)` for w13 weights, w2 weights, w13 scales, and w2 scales. Inferred from w13 copy-duration quantization, misses are about `4.4` per layer-step, implied hit rate is about `0.73`, and byte movement alone is about `18 ms/step` of the `20.1 ms/step` bucket. That makes `masked_row_copy` about `90%` C2C bandwidth cost and only about `2 ms/step` launch/empty-program overhead.

## Ranked next experiments

1. **Reduce miss bytes by improving hit rate / slot budget** — expected save still first-order, confidence medium. Evidence: inferred w13 quantization gives about `4.4` missed experts per layer-step, implied hit rate about `0.73`, and byte movement alone about `18 ms/step` of the `20.1 ms/step` `masked_row_copy` bucket.
2. **Amortize miss bytes across more tokens per step, especially MTP K=2** — expected save first-order if quality holds, confidence medium/low. Evidence: row-copy byte cost is per layer-step, so more accepted tokens per step amortize C2C movement; this matches `research/next-experiments-plan-2026-09-07.md`.
3. **Skip empty masks / coalesce row-copy launches** — expected save `≤2 ms/step`, confidence low/medium. Evidence: inferred byte movement is about 90% of `masked_row_copy`; only about 2.4% of w13 launches are below 10 us, so launch/empty-program overhead is not the 5–12 ms lever.
4. **Fuse or remove remaining non-GEMM `other` kernels** — expected save `1–2 ms/step`, confidence low/medium. Evidence: decode-only `other` is `7.18 ms/step`, including scalar gather at `0.50 ms/step` and MLA attention at `0.32 ms/step`.
5. **MoE GEMM shape/tuning pass for routed grouped GEMMs** — low priority for decode. Evidence: decode-path routed MoE GEMMs are only `5.38 ms/step`; the previous `22.1 ms/step` figure included prompt/prefill bypass rows.
6. **CUDA graph enablement** — not a current experiment. Evidence: decode is already graph-captured; about `4407` kernel instances/step are represented by about `189` eager kernel-launch APIs/step plus `3.06` CUDA graph launches/step.

Separate prefill/TTFT note: the bypass path costs about `7.8 ms/layer` across 75 cached layers, roughly `0.6 s` per short prompt here. That is worth a separate TTFT experiment, but it is not the current decode target.

## Limitations

- The trace has no layer id column; per-layer conclusions are inferred from launch ordering and config layer order.
- The full trace contains 143 inferred slot-cache steps, while the profile verdict/summary window uses 140 verification steps. The analysis explicitly uses the 140-step denominator for the published bucket table.
- Bytes moved for `masked_row_copy` are an upper bound because the trace does not expose the `mask` population per call.
- `live-receipts/nsys-buckets.json` was generated by the pre-fix regex and is preserved as-is. Use `analysis/decode-only-attribution.md` for the corrected decode-only bucket table; `analysis/corrected-buckets.json` is the superseded whole-capture corrected table.
- Aggregate GPU kernel time can exceed wall time because Nsight kernel summaries sum overlapped work across streams.
- The live receipt tree excludes large or sensitive raw artifacts listed in `RECEIPTS-EXCLUDED.md`.
