# E4b — make the counter free

Worker: **grok-4.6 / xai-oauth**. GLM stayed stopped-and-kept. `:30003` untouched. Never `docker rm`. v15 left **up** on `:30006`.

## D0 — zero-copy `topk_indices_buffer`

Not viable. `DeepseekV4Model.topk_indices_buffer` (model.py L424) is Indexer attention scratch `[max_num_batched_tokens, index_topk] int32`, comment "Reserved topk indices buffer for all Indexer layers to reuse." Passed into `attn`, not `DeepseekV4MoE`. Shared overwrite; after a step only the last indexer write is visible. Not MoE routed ids (`top_k=6`). `results/e4b/d0.json`.

## D1 — fixed-slice snap

`snap[40, T_max, 6].copy_(ids)` + `last_t[L]=T`. No `index_copy_`, no ring cursor. Drain samples the snap every `DRAIN_N` `execute_model` ticks (eager, not in-graph). GO-file park kept. Snap device 0.09 MiB. Staging 17.93 MiB (unchanged).

## Boots

| | container | bind | autotune |
|---|---|---|---|
| E4B freeze D1 | `dsv41-vllm-E4B-adaptive-EXP` (now `-D1-OVERHEAD-FAIL`) | ~8 min | 9ac7b387 Loaded 231 / 0 new / ~4.5 s |
| v15 control | `dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` (`docker start`) | 283 s | 9ac7b387 / 0 new |

## Freeze overhead (verdict instrument = knee)

Same-window E4B D1 freeze → v15. CAT_ORDER `prose structured code shell_ops tool_json`.

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| E4B D1 r1/r2 | 149.86 / 150.75 | 689.25 / 687.87 | 798.42 / 938.77 |
| v15 r1/r2 | 152.79 / 152.97 | 697.31 / 697.38 | 809.97 / 943.66 |
| mean | 150.31 vs 152.88 | 688.56 vs 697.34 | 868.60 vs 876.81 |
| overhead | **1.68%** | 1.26% | 0.94% |

Bar ≤ 1.0% C1. **FAIL.** Agent prose 160.3 vs 168.5. Weighted accept **59.6% both** — routing unchanged; tax is still extra CUDA-graph nodes (40 slice copies), not bandwidth of `T×6` int32. D2 (masked copy) has the same node count — not booted. D3: stop; adaptive placement needs an engine-side hook.

Host cgroup Δ: E4B +0.018 GiB, v15 +0.029 GiB. No growth.

## Not run (gated)

D2. Domain-shift JP/DE ↔ agent. 100-swap parity vs `results/e2c/parity-v15a.json`.

## Left running

`dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` on `:30006`. E4B stopped-and-kept as `-D1-OVERHEAD-FAIL`.

## E4b VERDICT: counter D1 overhead 1.68% · not run: D2 (same node count), unfreeze/shift/parity · left up: v15
