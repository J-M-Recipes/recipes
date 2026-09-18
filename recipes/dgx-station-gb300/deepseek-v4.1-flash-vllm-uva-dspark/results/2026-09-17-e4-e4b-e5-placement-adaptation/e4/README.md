# E4 — adaptive remap (EWMA + slow row swaps)

Worker: **grok-4.6 / xai-oauth**. GLM stayed stopped-and-kept. `:30003` untouched. Never `docker rm`. v15 left **up** on `:30006`.

## Dry-run (off-lane, before any boot)

Container `pin-e4-dry` (`--rm`). GPU 22 MiB. Cap 0.25. Real geometry E=384, rowmap layer-0 295/89.

Split vs full-E 100% bit-id at T=1/6/24/96. CUDA-graph T=6 100%. Five ordered swaps (hot-row → HBM staging → cold into hot → staging into cold → map flip) 100% bit-id before and after each. Misordered test hook (map flip, **no** row copies) diverged: bit=0.0135, max_abs=1.39e12. Staging 17.93 MiB (one HBM row, reused). `results/e4/dryrun.json`.

Ordering (documented in hook): copies complete on the main stream, then a 2-element write to the device `int32[384]` row map. Map **address** is fixed (CUDA graph); contents change between steps. Swaps enqueue from `GPUModelRunner.execute_model` / layer-0 eager `_invoke`. Never flip a layer mid-copy. Drain thread is parked on `<box-home>/pin-hot-experts/e4/GO` so it cannot `.item()` during graph capture.

## Boots

| | container | bind | autotune |
|---|---|---|---|
| E4 freeze | `dsv41-vllm-E4-adaptive-EXP` (now `-FREEZE-OVERHEAD-FAIL`) | ~8 min | 9ac7b387 Loaded 231 / 0 new / ~4 s |
| v15 control | `dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` (`docker start`) | 302 s | 9ac7b387 / 5 s |

Failed keep (not rm'd): `-GRAPHCAPTURE-FAILED`, `-GRAPHCAPTURE-FAILED2` (drain thread `.item()` during capture; fixed by GO-file park).

## Freeze overhead (verdict instrument = knee)

Same-window E4 freeze → v15. CAT_ORDER `prose structured code shell_ops tool_json`.

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| E4 freeze r1/r2 | 140.81 / 135.76 | 646.41 / 640.55 | 757.43 / 884.17 |
| v15 r1/r2 | 152.79 / 153.27 | 699.99 / 700.56 | 807.87 / 948.14 |
| mean | 138.29 vs 153.03 | 643.48 vs 700.28 | 820.80 vs 878.01 |
| overhead | **9.63%** | 8.11% | 6.52% |

Bar ≤ 1.0% C1. **FAIL.** Agent prose 145.5 vs 168.5. Weighted accept **59.6% both** — routing unchanged; the tax is the in-graph ring write (`index_copy_` + cursor math × 40 layers), the same class as E2a's 3.7% `scatter_add`. Did **not** unfreeze or run swaps.

Host cgroup Δ: E4 +0.027 GiB, v15 +0.047 GiB. No growth.

## Not run (gated)

Domain-shift JP/DE ↔ agent, 100-swap parity vs `results/e2c/parity-v15a.json`. Sampled 1-in-8 ring fallback not booted this window.

## Left running

`dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` on `:30006`. E4 freeze stopped-and-kept as `-FREEZE-OVERHEAD-FAIL`.

## E4 VERDICT: overhead 9.63% · swaps to plateau n/a · JP/DE n/a · agent recovery n/a · parity n/a (stopped before unfreeze) · left up: v15
