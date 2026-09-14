# 2026-09-14 — Agent-traffic slot remap, draft-routing correlation, hook scalar-fuse, K=2 re-gate → K2+fuse promoted

One morning, one box, all on the 256K profile (`sc13g`, 7,360 slots, 24 GiB bf16 KV, `max-num-seqs 1`, image `vllm-glm53-uva:v0.28.0-2cf0a691`). Follows the Sept-13 speed research (`../research/speed-research-2026-09-13.md`), which ranked five levers; this window measured all five. **Promoted at 10:35 CDT (James): K=2 + scalar-fuse hook v2 as DAILY.**

## Speed, fixed instrument (`windows/*/speed-reps.json`, `scripts/speed_reps.py` from Sept 13: 8 prompts × 512 tok × 3 scored reps, temp 0, C1)

| Lane (one axis off K1-256K DAILY args) | C1 median tok/s | vs mean(A,D) paired | wins |
|---|---:|---:|:-:|
| A idring2 = daily args + 379 MB inert routing ring | 51.01 | −0.8% vs D | |
| **B hook v2 `SLOT_CACHE_SCALAR_FUSE=1`, K=1** | **53.49** | **+4.58% [95% CI +4.42, +4.78]** | 8/8 |
| C `--max-num-batched-tokens 16384` | 51.29 | +0.19% | 8/8 |
| D control = daily args, stock hook (clean relaunch) | 51.40 | — | |
| **E hook v2 fuse + MTP K=2** | **54.69** | **+7.25% [95% CI +4.33, +10.44]** | 8/8 |

Per-prompt spread ≤0.04% on every lane. E vs B (K=2 increment on top of fuse): +2.55%, prose −2.5/−2.4, reason +10.0/0.0, code +5.5/+2.5, list +4.9, dialog +2.4 — same shape as the Sept-13 K=2 gate. B alone is below the +5% bar; E clears it.

## Quality receipts (`windows/{B,E}-k2-scalarfuse/`)

| | greedy vs D (20 prompts, `greedy_equiv.py`) | teacher-forced margins (`divergence_margin.py`, scored on the B server) |
|---|---|---|
| D self-repeat | 20/20 | — |
| **B fuse K1** | **20/20 byte-identical** | 0 sites |
| **E fuse K2** | 9/20 | 11 sites, mean \|Δlogp\| 1.023, max 4.50, K1-rank1 5 / K2-rank1 3; instrument repeat-identical max\|d\| = 0 |

E's margin summary is numerically identical to the Sept-13 K=2 fair gate (11 sites, 1.023, 4.50, 5/3). The fused hook is output-inert; the K=2 divergence is the already-adjudicated near-tie behaviour.

## What the hook v2 patch is (`scripts/slot_cache_hook_v2.py`, also `../patches/`)

`_cache_forward` used to update the three per-expert fp32 scalar arrays (`g1_scale_c`, `g1_alphas`, `g2_alphas`) with three `index_put_` calls plus two `.long()` casts per MoE layer per step. v2 replaces them with one Triton launch `masked_scalar_copy3` (fixed shape `NB = next_pow2(N)`, miss-masked, same sink-row semantics), enabled by `SLOT_CACHE_SCALAR_FUSE=1`; default off keeps the stock path. A2's estimate was ~0.5 ms/step; measured effect is ~0.8 ms/step (+2.1 tok/s). The v2 file also carries the optional `SLOT_CACHE_ID_RING` capture (off in the promoted lane).

## Falsified levers (measured, not modeled)

- **Agent-traffic slot remap.** `SLOT_CACHE_ID_RING` captured 96,504 complete decode steps of real Hermes tool-loop routing (32 tasks, `scripts/agent_corpus.py`, `ring_to_trace.py`). `alloc_slots.py` at 7,360: agent-trace LRU hit **0.6197 on the current map vs 0.6217 on the agent-optimized map** (`slots-7360.json`, 44 layers moved). +0.002 ≈ +0.1 tok/s. Dead. The workload's locality (0.62 sim ceiling vs 0.80 probe) is the limit, not the map.
- **Draft-routing prefetch.** 14,540 steps with the layer-78 MTP-block router captured alongside all 75 main layers (`draft_corr_per_layer.json`, `scripts/draft_corr.py`): draft(t)→main(t+1) overlap **0.0316 vs chance 0.0312**, 0/75 layers above 2× chance; draft names 3.1% of next-step LRU misses. Temporal same-layer 0.2524 reproduces the trace figure. The +23–37 tok/s prefetch family is closed on this model.
- **`--max-num-batched-tokens 16384`.** +0.19% decode. Prefill tok/s was not captured by the driver (field-name mismatch); needle 32K/64K/128K single passed on all lanes.

## Promotion / rollback

- DAILY: `glm53-big-sc13g-mtp2-ctx256k-K2fuse-DAILY-20260914` (daily args + `SLOT_CACHE_HOOK=/w/slot_cache_hook_v2.py SLOT_CACHE_SCALAR_FUSE=1 SLOT_CACHE_ID_RING=0`, `num_speculative_tokens: 2`). Launcher: `scripts/launch-k2-scalarfuse.sh`.
- Rollback: `docker start glm53-big-sc13g-mtp-ctx256k-K1-daily-keep-pre-fuse-20260913` (the Sept-13 K1 daily, stopped+kept). B (`…-scalarfuse-20260914`, K1+fuse) also kept.
- Sequence and timestamps: `LEDGER.md` on the Station, 2026-09-14 04:5x–10:3x CDT. Interrupted 08:47–09:20 for an electrician shutdown (clean `docker stop` + `shutdown -h`); phase 3 resumed after reboot.

`promotion_authorized=true` for this result only, by James, 2026-09-14 10:3x CDT. Written by Milo on Claude (claude-fable-5-1, extended thinking).
