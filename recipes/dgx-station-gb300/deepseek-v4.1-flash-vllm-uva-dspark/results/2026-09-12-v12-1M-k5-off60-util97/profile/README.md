# Decode-step breakdown — v12, DSpark k=5, torch profiler (2026-09-12 06:35 CDT)

Boot identical to v12 plus `--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=… --profiler-config.max_iterations=60`; hash hit (7.5-min boot); knee C1 on the profiler boot 89.5 (overhead negligible at C1). Three captures via `/start_profile` … `/stop_profile`, ~150 decode tokens each. Kernel time grouped by role with `prof_breakdown.py` (the `bmm_MxE4m3_MxE2m1…` kernels are the FlashInfer TRT-LLM fused-MoE expert GEMMs, i.e. weight streaming).

| role | prose C1 | shell C1 | prose C8 |
|---|---|---|---|
| **MoE expert GEMMs** (weight-streaming) | **64%** | 65% | **88%** |
| attention (sparse MLA + indexer, incl. quantize) | 17% | 17% | 6% |
| dense GEMMs (deep_gemm, nvjet) | 5% | 6% | 3% |
| norm / mhc / elementwise / memcpy | 14% | 12% | 3% |
| GPU busy | ~100% | ~100% | 97% |

Read with the all-HBM two-station reference for this model (141 tok/s C1, ~12 ms step, same attention): the Grace expert fetch is **~9 ms of a ~24 ms C1 step**. At C8 the batch touches most experts and expert streaming is 88% of the step — the C16 flattening; no residency scheme moves that.

Raw traces (3 × ~10 MB gz, Perfetto-loadable) are kept off-repo on the station under `prof-PROF-v12/`; ask and they will be shared.
