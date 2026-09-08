# E1 v2 decode-only attribution correction

Correction date: September 8, 2026.

## Method

The first E1 v2 Nsight cut divided the whole capture's aggregate GPU kernel time by the 140 verification steps. That was a methodology limitation: the capture includes prompt/prefill kernels, and for this probe the prompt rows are separable by instance count.

Profile scope:

- 4 sequential requests;
- about 50 prompt tokens each;
- 64 completion tokens each, 256 completion tokens total;
- 140 verification steps in the verdict window (143 inferred slot-cache steps in the trace including 3 warm steps);
- MTP K=1;
- 75 slot-cached MoE layers;
- `BYPASS=16`, so prompts longer than 16 tokens take the bypass path that streams all expert weights over C2C.

Instance-count separation:

- rows with `Instances == 300` (`4 requests * 75 cached layers`) and prompt-shape MoE GEMM names are prefill bypass rows:
  - `bmm_E2m1_E2m1E2m1_*_t128x32x512u2_s5_*`: 1563.5 ms total, 5211.8 us average;
  - `bmm_Bfloat16_E2m1E2m1_*_t128x32x512u2_s4_*`: 780.1 ms total, 2600.5 us average.
- two `Instances == 4` dense prompt rows are also prefill/TTFT work: 5.1 ms and 2.6 ms.
- total prefill-attributed aggregate GPU time: 2351.4 ms.
- decode-path routed MoE GEMMs are the `Instances == 10725` routed rows: 479.7 ms + 273.7 ms = 753.4 ms total = 5.38 ms/step.

Planned analysis fix: either exclude `Instances == requests * cached_layers` prompt rows from the decode denominator or trim the capture window to decode-only. This correction does not rewrite the preserved live runner receipt `live-receipts/nsys-buckets.json`.

## Reproducible script output

Command:

```bash
.venv/bin/python recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-08-e1-v2-live/analysis/decode_only_buckets.py
```

Output:

```text
source: /Users/jamesmeadlock/hermes/jm-recipes/recipes/recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-08-e1-v2-live/live-receipts/e1_cuda_gpu_kern_sum.csv
rows: 108
prefill_rows: 4 prefill_gpu_ms: 2351.4
decode_kernel_instances_per_step: 4407.3
decode_wall_ms_per_step: 40.06
decode_gpu_ms_per_step: 43.13
residual_wall_ms: 2610 (652 ms/request)

| bucket | ms/step | % decode GPU | instances | rows |
|---|---:|---:|---:|---:|
| masked_row_copy | 20.10 | 46.6% | 42,900 | 1 |
| dense_gemm | 9.41 | 21.8% | 124,007 | 27 |
| other | 7.18 | 16.6% | 417,934 | 73 |
| routed_moe | 5.38 | 12.5% | 21,450 | 2 |
| fused_bookkeeping | 1.05 | 2.4% | 10,725 | 1 |
```

## Corrected decode-only table

| bucket | ms/step | % decode GPU | note |
|---|---:|---:|---|
| `masked_row_copy` | 20.10 | 46.6% | 42,900 total instances; last 140 inferred decode steps give 300 launches/step. |
| dense GEMM | 9.41 | 21.8% | Dense GEMM bucket after separating prompt bypass work. |
| other | 7.18 | 16.6% | Includes scalar gather at about 0.50 ms/step and MLA attention at about 0.32 ms/step. |
| routed MoE GEMM | 5.38 | 12.5% | Decode-path `Instances == 10725` routed rows only. |
| fused bookkeeping | 1.05 | 2.4% | Still below the opportunity gate with scalar gather. |

Decode wall from the profiled probe is `256 tokens / 45.648 tok/s / 140 = 40.06 ms/step`, reported as 40.1 ms/step. Decode aggregate GPU is `(8389.2 - 2351.4) / 140 = 43.13 ms/step`, reported as 43.1 ms/step. The GPU aggregate still exceeds decode wall, so the decode path remains GPU-bound; kernel summaries are aggregate work and can exceed wall when streams overlap.

## Residual wall reconciliation

The full profiled wall was 8.218 s. The decode wall reconstructed from throughput is 5.608 s, leaving `8.218 - 5.608 = 2.61 s` for prefill plus TTFT across four prompts, or about 0.65 s per prompt.

That residual matches the bypass rows: the two large prompt MoE rows average about `5.2 ms/layer + 2.6 ms/layer = 7.8 ms/layer`; across 75 cached layers this is about 0.59 s per prompt before smaller overheads. This is consistent with the C1 bench mean TTFT range of about 0.61–0.66 s.

## CUDA graph correction

A second same-day correction: the first analysis also misread `backend=eager` in the vLLM compilation config as eager execution. It does not mean no CUDA graph replay. In vLLM compilation mode 3, `backend=eager` disables Inductor; piecewise/full CUDA graphs can still be active. The container log line `Inductor compilation was disabled by user settings` is consistent with that read.

Measured from receipts: decode kernel rows have about 4407 kernel instances/step, while the CUDA API trace has about 189 kernel-launch API calls/step plus 3.06 `cudaGraphLaunch_v10000` calls/step. Therefore roughly 4200 kernels/step are replayed inside about three CUDA graph launches. Decode is graph-captured; the 189 eager launches/step are the non-captured remainder. The earlier 5.5 ms/step launch-API number is mostly overlapped and is not a first-order optimization lever.

## Inferred row-copy byte model

The following is inferred, not directly measured by the Nsight summary. From the w13 copy-duration quantization, 32.8 us per missed row at 383 GB/s implies about 4.4 missed experts per layer-step and an implied hit rate of about 0.73, consistent with E0's reconstructed about 0.75. The byte cost alone is approximately `4.4 * 21.2 MB / 383 GB/s = 0.24 ms/layer`, or about 18 ms/step across 75 cached layers. That accounts for about 90% of the 20.1 ms/step `masked_row_copy` bucket; only about 2 ms/step is launch/empty-program overhead. Only 2.4% of w13 launches are below 10 us; the older about-15% near-empty observation was dominated by the small scale-tensor launches.

Consequence: coalescing launches or skipping empty masks is no longer a 5-12 ms/step lever. The remaining likely levers are fewer miss bytes through hit-rate/slot-budget changes and amortizing misses across more tokens per step, especially MTP K=2, consistent with `research/next-experiments-plan-2026-09-07.md`.

## What changed in the conclusions

Unchanged:

- the E1 v2 contract verdict remains inconclusive by design;
- the bookkeeping + gather hypothesis remains falsified (`1.56 ms/step < 2.0 ms/step`);
- `masked_row_copy` still launches about 300 times/step, with p50 34 us, p95 214 us, and max 519 us;
- decode remains GPU-bound and graph-captured;
- profiler slowdown remains about 1.6%;
- comparator benches and API attribution counts are unchanged.

Changed:

- the headline is no longer GPU-bound at about 59 ms/step. The corrected decode-only view is about 40.1 ms/step wall and 43.1 ms/step aggregate GPU.
- `masked_row_copy` is about half of decode (`20.10 ms/step`, 46.6% of decode GPU), not about a third.
- MoE GEMM tuning drops in priority: decode-path routed MoE GEMMs are only 5.38 ms/step, not 22.1 ms/step.
- CUDA graph enablement is not the next lever: decode is already graph-captured, and the 189 eager launches/step are only the non-captured remainder.
- launch coalescing / empty-mask skipping drops to a low/medium-confidence `<=2 ms/step` lever; inferred byte movement is about 90% of `masked_row_copy`.
- the bypass prefill path is a separate TTFT issue: prompt bypass costs about 7.8 ms/layer, roughly 0.6 s per short prompt in this probe, but that is not the current decode target.
