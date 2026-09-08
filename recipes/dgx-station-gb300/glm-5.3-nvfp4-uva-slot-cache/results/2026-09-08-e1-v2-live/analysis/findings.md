# E1 v2 Nsight findings — GLM-5.3 NVFP4, GB300, K=1

## Headline numbers

- Wall: `8.217965s / 140 = 58.700 ms/step`.
- Aggregate GPU kernels: `8389.216 ms / 140 = 59.923 ms/step` (`1.021x` wall; aggregate exceeds wall by `1.223 ms/step`, so there is little/no idle exposed by this window; decode is GPU-bound, with caveat that kernel sums are aggregate-overlap not critical path).
- Profiler overhead: `1.61%` throughput drop from unprofiled median decode `46.383` tok/s to profiled `45.648` tok/s.
- Previous `other_gpu` was `37.953 ms/step`; corrected bucketing leaves `6.416 ms/step`, attributing `31.537 ms/step = `83.09%` of the old bucket.

## Corrected GPU buckets

| bucket | ms/step | % aggregate GPU | instances | rows |
|---|---:|---:|---:|---:|
| routed_moe | 22.122 | 36.92% | 22,050 | 4 |
| masked_row_copy | 20.098 | 33.54% | 42,900 | 1 |
| dense_gemm | 9.415 | 15.71% | 124,007 | 27 |
| fused_bookkeeping | 1.054 | 1.76% | 10,725 | 1 |
| scalar_gather | 0.504 | 0.84% | 33,991 | 7 |
| mla_attention | 0.315 | 0.53% | 23,226 | 2 |
| mtp_verify | 0.000 | 0.00% | 0 | 0 |
| memcpy/memset | 0.000 | 0.00% | 0 | 0 |
| other | 6.416 | 10.71% | 360,725 | 66 |

## Interpretation

- Falsified hypothesis: `fused_bookkeeping + scalar_gather = 1.558 ms/step`, below the `2.0 ms/step` gate.
- Actual dominant single bucket: `routed_moe` grouped NVFP4 GEMMs at `22.122 ms/step`, just above `masked_row_copy` at `20.098 ms/step`.
- Bigger lever depends on scope:
  - MoE GEMM only (`routed_moe`) is `22.122 ms/step`, ~`1.10x` masked copy.
  - All GEMM buckets (`routed_moe + dense_gemm`) are `31.537 ms/step`, clearly larger than masked copy.
- CUDA API trace: total CUDA API duration is `36.671 ms/step`, dominated by `cudaEventSynchronize` waits (`29.867 ms/step`). Kernel launch API time is `5.516 ms/step` over `189.0` eager launch APIs/step.
- CUDA graphs: `cudaGraphLaunch_v10000` appears `429` times (`3.06/step`) vs `cudaLaunchKernel` `17329` times (`123.78/step`). With launcher compilation-config mode 3 backend eager, this means no full-decode CUDA graph replay; most work is still eagerly launched.

## Ranked next experiments

1. **Reduce/eliminate slot-cache row copies for cached hits/misses** — expected save `5-12 ms/step`, confidence medium. Evidence: `masked_row_copy` is `20.1 ms/step`; many calls are small/empty, but weight-row copies are still ~150us/75us modes. Test coalescing copies, copying only actually missed experts, persistent resident hot experts, or async prefetch/overlap.
2. **MoE GEMM shape/tuning pass for routed grouped GEMMs** — expected save `3-8 ms/step`, confidence medium. Evidence: routed NVFP4 grouped GEMMs are the largest single bucket (`22.1 ms/step`) and the top two long-shape rows alone are `16.7 ms/step`.
3. **Enable/capture full decode CUDA graphs or reduce eager launch count** — expected save `1-3 ms/step`, confidence medium. Evidence: launch APIs excluding graphs are `5.52 ms/step` and ~`189` calls/step; mode 3 eager is not graph replaying the decode.
4. **Fuse or remove remaining non-GEMM `other` kernels after corrected bucketing** — expected save `1-2 ms/step`, confidence low/medium. Evidence: remaining `other` is `6.4 ms/step`, spread across 66 rows; top entries include sparse attention, index, norm, routing, and copies, so wins require targeted fusion.
5. **Validate cached-layer allocation / early-layer slot pressure** — expected save `1-4 ms/step`, confidence low. Evidence: inferred layer pattern has early layers L3-L7 slowest and configured slots correlate with copy time; try fewer cached layers, different slot budget, or keeping early hot layers resident.

## Input SHA256 audit

```json
{
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/e1_cuda_api_trace.csv": "e737f4ed0512d1302012faff5530e39708df78ce419f6e0a52fba55c8d688e77",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/e1_cuda_gpu_kern_sum.csv": "cd768d4db40374606b08db01968a5ff796d44467ed4a2ca9cf0c3793c4ddaab7",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/e1_cuda_kern_exec_sum.csv": "3ee79cc2e6b4adccb097664dee578e68de22ae276a7df28ae6c3d631c6ff2e70",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/executed-source/configs/slots-5792-ctx512k.json": "73962276498626af16baaa7c8737384960aff8718d254e66ab92610570c9743c",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/executed-source/patches/slot_cache_hook.py": "2e7fb1e5dd621eb465add5a5766fe01dd0a7bc2872ecdb70027b251a43cfdc46",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/executed-source/scripts/nsys_bucket.py": "e6babffd414c4ae9b2ecd9409fe9b3c653bc4f68ba7808297b2f49c6fae9a7da",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/nsys-buckets.json": "a0e5986b9881c5641ac495a0875f5b2067aec886cb847eee64b2c47fbb7a459e",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/profile-wall-seconds.txt": "26fecbc4d007385d0a19e2b5c4a8ed270558cbaa8215ae8dbfc18b5ebb921434",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/profiled-probe.json": "a5dec44f719b01fecb4123bd3aa7598a9731ce95bf06f8733b694e3ab162a273",
  "/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/unprofiled-probe.json": "8be49bd51711c80b58caba0afdbbb6e4f86fa1412ff4ae7189ddde5d29452939",
  "milo@192.168.1.9:/home/milo/e1-window-20260908-v2/receipts/e1-profile.sqlite": "3b7760d5a04cd68cc0c9dd752a2372a0ef19c33cd727e1582933f100dd92ded0",
  "milo@192.168.1.9:/home/milo/e1-window-20260908-v2/receipts/e1_cuda_gpu_trace.csv": "0ba12baaf22db45d3efa39fe4cf6f791faa6029e7c2f0e0936fc38266017bb6a"
}
```
