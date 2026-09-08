# E1 v2 live profile — 2026-09-08

Final state: **profile collected; contract verdict INCONCLUSIVE by design; offline analysis falsified the bookkeeping hypothesis.**

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

This profile is GPU-bound at the window level: wall is `8217.965 ms / 140 = 58.699750 ms/step`; aggregate GPU kernels are `8389.216241 ms / 140 = 59.922973 ms/step`, exceeding wall by `1.223223 ms/step` because kernel summaries aggregate overlapped GPU work.

## Corrected GPU buckets

Corrected buckets from `analysis/corrected-buckets.json`:

| bucket | ms/step | % aggregate GPU | instances | rows |
|---|---:|---:|---:|---:|
| routed_moe | 22.121659 | 36.916825% | 22,050 | 4 |
| masked_row_copy | 20.097824 | 33.539430% | 42,900 | 1 |
| dense_gemm | 9.414931 | 15.711723% | 124,007 | 27 |
| fused_bookkeeping | 1.053915 | 1.758784% | 10,725 | 1 |
| scalar_gather | 0.503603 | 0.840418% | 33,991 | 7 |
| mla_attention | 0.315024 | 0.525715% | 23,226 | 2 |
| mtp_verify | 0.000000 | 0.000000% | 0 | 0 |
| memcpy/memset | 0.000000 | 0.000000% | 0 | 0 |
| other | 6.416016 | 10.707106% | 360,725 | 66 |

The pre-fix live bucket file had `other_gpu = 37.952606 ms/step`. Corrected bucketing reattributes `31.536590 ms/step` of that bucket to `routed_moe` and `dense_gemm`, leaving `other = 6.416016 ms/step`.

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

Per-call distribution over all rows: min `0.992 us`, p50 `34.144 us`, p95 `214.496 us`, p99 `323.616 us`, max `519.168 us`. About 14.16% of calls in the last 140 inferred steps are below 10 us. The launch pattern is fixed `GrdY=(768, 384, 96, 48)` for w13 weights, w2 weights, w13 scales, and w2 scales. Early inferred cached layers L3–L7 are slowest.

## Ranked next experiments

1. **Reduce or eliminate slot-cache row copies for cached hits/misses** — expected save `5–12 ms/step`, confidence medium. Evidence: `masked_row_copy` is `20.1 ms/step`; many calls are small or empty, but weight-row copy modes are still about 150 us and 75 us.
2. **MoE GEMM shape/tuning pass for routed grouped GEMMs** — expected save `3–8 ms/step`, confidence medium. Evidence: `routed_moe` is the largest single bucket at `22.1 ms/step`; the top two long-shape rows are `16.7 ms/step`.
3. **Enable/capture full decode CUDA graphs or reduce eager launch count** — expected save `1–3 ms/step`, confidence medium. Evidence: non-graph launch APIs are `5.52 ms/step` and about `189` calls/step; mode 3 eager is not full decode graph replay.
4. **Fuse or remove remaining non-GEMM `other` kernels** — expected save `1–2 ms/step`, confidence low/medium. Evidence: remaining `other` is `6.4 ms/step` across 66 rows.
5. **Validate cached-layer allocation / early-layer slot pressure** — expected save `1–4 ms/step`, confidence low. Evidence: inferred early layers L3–L7 are slowest and configured slots correlate with copy time.

## Limitations

- The trace has no layer id column; per-layer conclusions are inferred from launch ordering and config layer order.
- The full trace contains 143 inferred slot-cache steps, while the profile verdict/summary window uses 140 verification steps. The analysis explicitly uses the 140-step denominator for the published bucket table.
- Bytes moved for `masked_row_copy` are an upper bound because the trace does not expose the `mask` population per call.
- `live-receipts/nsys-buckets.json` was generated by the pre-fix regex and is preserved as-is. Use `analysis/corrected-buckets.json` for the corrected bucket table.
- Aggregate GPU kernel time can exceed wall time because Nsight kernel summaries sum overlapped work across streams.
- The live receipt tree excludes large or sensitive raw artifacts listed in `RECEIPTS-EXCLUDED.md`.
