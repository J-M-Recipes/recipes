# GLM-5.3-big on one GB300 — where it stands, what is closed, what is next (2026-09-15)

Top-down review by Milo, 2026-09-15. Sources: this recipe's results bundles (2026-09-05 → 2026-09-14), `speed-research-2026-09-13-synthesis.md`, the Nsight E1 v2 profile, and the live Station state on the morning of 2026-09-15. Numbers are measured unless marked *modeled*.

## 1. Current state

| item | value |
|---|---|
| Daily profile | K=2 MTP + hook v2 scalar-fuse, 256K ctx, 24 GiB bf16 KV, 7,360 slots, `--max-num-seqs 1` |
| Decode, C1 | **54.7 tok/s** (8 prompts × 512, median of medians); V1 offload-only baseline 33.8 (+62%) |
| Quality | teacher-forced logprobs vs V1, 3,071 tokens, max abs dlogp = 0.0 (byte-identical); needle 18/18 to 211K |
| Step budget (Nsight, K=1, 512K) | 40.1 ms/step: **18 ms C2C miss bytes** (masked_row_copy, 46.6%) · 9.4 ms dense GEMM (21.8%) · 5.4 ms routed MoE GEMM (12.5%) · ~7 ms other · ~1 ms bookkeeping |
| Live hit rate | probes 0.61 (K1) / agent traffic 0.41–0.44 (K2); exact full-trace LRU ceiling at this budget 0.802 |
| Station on 2026-09-15 08:30 CDT | `:30001` **dark** — daily container stopped 2026-09-14 ~12:00 CDT for the DSV4.1 lane and not restarted (no-restore-on-done rule). Restart is `docker start glm53-big-sc13g-mtp2-ctx256k-K2fuse-DAILY-20260914` after `nvidia-smi` shows < 2 GiB. |

## 2. Corrections to earlier statements in this recipe

Data overturned two claims that were written into the README on 2026-09-13; they are struck in place there and recorded here.

1. **"LRU 0.719 at 7,360 slots" was a `T//8000` subsample.** Exact full-trace replay of the same 71,210-token corpus at the same budget gives **hit 0.802 / 1.584 misses per layer-step**. The conclusion (cache beats any static allocation; oracle pin 0.633) stands; the ceiling is higher than published, so the live-vs-ceiling gap (0.41–0.61 live vs 0.80) is larger than the table implied.
2. **"The remaining signal is one step ahead (MTP draft routing → async miss fill)" is false as a mechanism.** GLM-5.3's MTP block is a separate one-layer MoE with its own fp32 gate and 256 experts (`glm4_moe_mtp.py`); the drafter/target `topk_indices_buffer` sharing is for the DSA indexer, not MoE routing. Measured on 14,540 agent steps: draft(t)→main(t+1) expert overlap **0.0316 vs chance 0.0312**, 0/75 layers above 2× chance. Previous-step same-layer prefetch has **0.000 miss coverage** (LRU already holds it). The only surviving path to the *modeled* +23–37 tok/s lookahead bound is a **trained** route predictor, which is a research project, not a serving lever.

## 3. Why the single-stream lane is at its floor

Every lever in the miss-byte family has a receipt and none moved the number:

| lever | measured | receipt |
|---|---|---|
| row-copy kernel rewrite | already 348–367 GB/s = 87–105% of effective C2C peak; bytes are the cost, not the kernel | E1 v2 Nsight, `speed-research-2026-09-13.md` |
| agent-traffic slot remap (same 7,360) | +0.002 hit on 96,504 captured agent steps | `results/2026-09-14-agent-remap-draftcorr-scalarfuse-k2/` |
| MTP-draft routing prefetch | at chance (above) | same |
| previous-step prefetch | 0.000 coverage | `speed-research-2026-09-13-synthesis.md` (exact replay) |
| offline static reallocation / adjacent-layer predictor | 8.4% precision vs 50% gate | `results/2026-09-07-offline-cache-prefetch/` |
| demand-fill DMA (PR #1) | 3.28% slower than Triton eager | `results/2026-09-07-dma-demand-fill/` |
| DFlash2 draft over UVA | accepted length 1.57 vs 3.0 gate | `results/2026-09-07-dflash2-uva/` |
| KV off HBM to buy slots (*modeled*) | +0.7 tok/s at 30–60K, wash at 150K, −0.4 at 256K | synthesis, lever 3 |
| `--max-num-batched-tokens 16384` | +0.19% decode | 2026-09-14 bundle |
| launch coalescing | capped by ~2.1 ms/step overhead; ≤ +3 tok/s best case | A2 analysis |

What remains on C1 is a set of ≤5% items: K=3 (K=2 already needed the fuse to clear +5%), vLLM 0.29 CUDA routing for GlmMoeDsa (PR #52861; MRV2 default breaks the FFI router seam, full re-gate), engine upgrades generally. None is worth a window unless the hook maintenance cost (item C below) forces an engine move anyway.

The physics: at batch 1 the dense path is HBM-bandwidth-bound, the miss path is C2C-bandwidth-bound and already at peak, and LRU is at the workload's locality ceiling. There is no ≥1.5× single-stream lever left on this model with this offload design.

## 4. Where a significant gain can still come from

Ranked by cost. None of these is authorized by this document; each is one axis, measurement-only, per `glm53-gb300-uva-experiments`.

### A. Concurrency on the big lane (one boot, config-only) — untested

`--max-num-seqs 1` today; batch > 1 is not validated on the 256K/7,360-slot profile. Miss bytes are paid per *step*, not per token, and the union of routed experts across N streams grows sublinearly (top-8 of 256, cross-layer at chance, but same-layer temporal overlap 0.25–0.27). The V1 offload-only baseline already showed C4 aggregate 57.7 vs C1 33.8 (1.7×) with no slot cache. Expected: **~2–2.5× aggregate at C4** with per-stream latency near C1 — but this is a *modeled* guess from the V1 ratio; the slot cache could go either way (shared working set helps, K=2 verify traffic hurts).

What to run: `MAX_NUM_SEQS=4`, `--kv-cache-memory` unchanged (24 GiB covers 4 × 64K comfortably; document that 4 × 256K does not fit and the scheduler will queue). Instruments: the session-ordered agent replay from the DSV4.1 lane (`deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-14-round6-*/`, ±1% run-to-run at 300 turns) plus C1 `speed_reps.py` to prove C1 did not regress. Gate: greedy 20/20 self-repeat at C1 unchanged; report C1/C2/C4 aggregate and per-stream. This is the lever that matters for subagent fan-out, not for a single chat turn.

### B. GLM-5.3-Flash is the 4× answer (separate recipe, separate lane)

Same box, all-HBM: the Flash DFlash2 lane measured 252 tok/s C1 at 0.39 acceptance; a third party on the same silicon class reports 305–470 at 0.54–0.67. The 2026-09-14 Flash research (`~/hermes/glm-flash-research-20260914/SYNTHESIS.md`) attributes the gap to a drafter checkpoint two revisions behind HF main plus three merged SGLang verify-path fixes after the pinned nightly. That is the Flash recipe's work, not this one's; it is listed here only because it is the honest answer to "how do I get a big speed gain on this box" — pick the model that fits HBM. The 744B's value is capability, and this recipe should stop chasing tok/s once A is answered.

### C. Retire the fork: evaluate upstream vLLM expert offload (weeks)

vLLM issue #38256 (dynamic MoE expert offload: pinned-host experts, fixed GPU expert cache, LFRU, cross-layer prediction, "zero-copy mode" on coherent platforms) is a generalized version of this recipe's 27 KB `slot_cache_hook.py`. The project: boot it on GLM-5.3-big at the same HBM budget, compare to 54.7 C1 with the same instruments, and contribute what this recipe already proved on Grace-C2C (cross-layer prediction is at chance on this model; LRU beats any static map; row copy is at C2C peak; `cudaHostRegister` path 340 GB/s vs 90 GB/s for GPU-page-table mappings). If it matches, the daily moves to an unpatched image and the hook, the FFI router seam, the sitecustomize mount, and the exact-hash review envelope all go away. Not a speed lever; a maintenance and OSS-contribution lever. Do after A, and only if the big model remains a daily driver rather than a capability reserve.

### Not next

- Trained expert predictor + async double-buffer fills: the only route to the +23–37 tok/s bound, but a research project with a low prior (adjacent-layer predictor hit 8.4% precision). Park.
- vLLM 0.29 upgrade for its own sake: engine-refresh axis, full re-gate, FFI router revalidation. Do it only as part of C.
- More instrumentation on this recipe. The quiescent-snapshot package and review envelope are complete for what they were built for; adding more does not change a number.

## 5. Recommended order

1. Restart the daily on `:30001` (needs James's word; restore rule).
2. Flash lane: drafter swap → nightly rebuild → per-class acceptance (Flash recipe, new session).
3. A (concurrency) in a window when the big model is not in use.
4. C only if the big model stays daily.

Written by Milo on Claude (claude-fable-5-1). No Station changes were made for this note.
