# 2026-09-13 — K=2 fair gate (teacher-forced margins), usable-context ladder, routing-trace analysis

Three questions, one box, one afternoon, all on the 256K daily profile (`sc13g`, 7,360 slots, 24 GiB bf16 KV, `max-num-seqs=1`, image `vllm-glm53-uva:v0.28.0-2cf0a691` / `sha256:61fc8a896b0a…`).

**Verdicts**

1. **K=2 is quality-equivalent to K=1 by the model's own scoring, and the greedy-identical gate was the wrong instrument.** Every one of the 11 K1-vs-K2 divergence sites scores byte-identically on the K1 server and on the K2 server (max |Δlogp| between servers = **0.0**). The two lanes share prefill numerics exactly; they only differ in which token the decode path picks at near-ties. K1's choice is rank-1 at 5 sites, K2's at 3, neither at 3.
2. **K=2 is +3.2% faster at 256K (paired, 95% bootstrap CI +0.5…+6.1, 6/8 prompt wins) — real, but below the +5% promotion bar.** Not promoted. Reopened: it is the lane that benefits most from any future miss reduction.
3. **256K declared is fully usable.** Single / multi-depth / distractor needle variants pass at every rung through 240K target (211K actual prompt tokens); prefill holds ~3.4–3.5K tok/s flat; 62 s wall at the top rung.
4. **On this model, LRU beats explicit expert allocation.** Static oracle pin at the live slot budget: 0.633 mean hit. LRU: 0.719. Cross-layer expert overlap is at random-chance (0.031 vs 0.031 baseline) — "route L, prefetch L+1" has nothing to work with. Temporal overlap step→step is 0.27 (vs 0.03 random) — that is the signal LRU harvests, and it is the signal a one-step-ahead (MTP-draft) prefetch would use.

## 1. Fair quality gate: teacher-forced divergence margins (`*/margin-on-K*.json`, `scripts/divergence_margin.py`)

For each of the 11 prompts where K1 and K2 greedy text differs, find the first divergent *token*, teacher-force the shared prefix via `/v1/completions` (`echo=true, logprobs=1, max_tokens=1`, `chat_template_kwargs.reasoning_effort=low` to match generation), and read the model's logprob of K1's token vs K2's token at that position. Instrument proof: same sequence scored twice → identical (max |Δ| = 0) on both servers.

| | scored on K1 server | scored on K2 server |
|---|---|---|
| sites | 11 | 11 |
| mean \|Δlogp\| (K1 tok − K2 tok) | 1.023 | 1.023 |
| max \|Δlogp\| | 4.50 | 4.50 |
| K1 token rank-1 / K2 token rank-1 | 5 / 3 | 5 / 3 |
| max \|logp(K1srv) − logp(K2srv)\| across all 22 scored tokens | **0.0** | |

Self-fidelity (`selfcheck-*.log`, `scripts/selfcheck.py`): fraction of a lane's own greedy tokens that are rank-1 when its text is teacher-forced. K1 text: **2866/3030 = 0.946** on both servers. K2 text: **2834/3006 = 0.943** on both servers. Symmetric; neither lane is "the model's preference." The 5–6% non-rank-1 tokens are prefill-vs-decode numerics (chunked prefill uses different GEMM shapes than 1-token decode), identical for both lanes.

Determinism: K2 greedy on this relaunch vs the earlier K2 window r2 → **20/20** identical (`k2-256k-gate/greedy.log`). Third self-repeat pass today.

**Read:** the September 9 "K2 fails greedy 9/20" and today's reproduction of it measure kernel-shape equivalence (verifying 3-token blocks vs 2 changes reduction order), not quality. Where the two lanes disagree, the model itself rates them as near-ties or splits favour evenly. The blog's "K=2 stays closed" verdict is retracted on quality grounds; it remains unpromoted on speed grounds only.

## 2. Speed, fixed instrument (`*/speed-reps.json`, `scripts/speed_reps.py`)

8 prompts (2 prose, 2 code, 2 reason, 1 list, 1 dialog) × 512 tokens, temp 0, streamed; rep 0 warm and discarded, 3 scored reps; per-prompt median and spread; overall = median of medians. Both lanes alone on the GPU, run back to back (K1 daily stopped+kept, K2 relaunched from its kept container).

| | K1-256K | K2-256K |
|---|---:|---:|
| C1 median tok/s | **51.50** | **52.92** |
| max per-prompt spread | 0.48% | 0.04% |
| TTFT median (50–60 tok prompts) | 0.498 s | 0.507 s |
| paired Δ per prompt | prose −2.1, −1.9 · code +6.1, +3.2 · reason +10.7, +0.6 · list +5.6 · dialog +3.1 |
| paired mean / 95% bootstrap CI | **+3.17%** / [+0.52, +6.08] |
| K2 wins | 6 / 8 |
| live slot hit during window | ~0.55 (K1) | ~0.44 (K2) |

K2 gains accepted tokens per step (2.2–2.8 of max 3 vs 1.8 of max 2) but routes ~40% more experts per step and pays for it in misses; prose has the lowest acceptance and nets negative. Bar for promotion is +5% on this instrument; the CI lower bound clears zero, not five.

## 3. Usable-context ladder, K1-256K (`k1-256k-gate/needle-ladder.jsonl`, `scripts/needle_ladder.py`)

Random-word filler (defeats prefix cache), three variants per rung, one retry with a fresh seed on failure, stop on a failed rung. Served through `/v1/completions` with the chat render pre-closed `<think></think>` (this template has no think-off switch; only low/high/max effort).

| target | prompt tokens | single | multi (5 keys @10/30/50/70/90%, ordered) | distractor (blue vs 4 same-prefix decoys) | wall (s) | prefill tok/s (upper) |
|---:|---:|:-:|:-:|:-:|---:|---:|
| 8K | 7.1K | PASS | PASS | PASS | 2.3 | 3,100 |
| 32K | 28.3K | PASS | PASS | PASS | 8.6 | 3,300 |
| 64K | 56.5K | PASS | PASS | PASS | 15.9 | 3,550 |
| 128K | 112.6K | PASS | PASS | PASS | 31.9 | 3,528 |
| 192K | 169.2K | PASS | PASS | PASS | 48.5 | 3,486 |
| 240K | 211.3K | PASS | PASS | PASS | 61.2 | 3,452 |

18/18, zero retries. Prefill is flat across the range on the offloaded recipe — the bypass path reads non-resident experts per chunk, and that cost does not grow with position. The 2K smoke rung (not in the file) was 1.2–1.3K tok/s: fixed per-request overhead dominates below ~8K.

**Read:** 256K declared is not paying for context nobody can use. That reverses the earlier framing of "just declare less context" as the free speed lever; the right lever is decoupling KV from expert slots (KV offload / admission patch), so 512K–1M declared can keep 256K's slot budget.

## 4. Routing-trace analysis (`trace-analysis/`, `scripts/trace_analysis.py`)

Offline, on the frozen `r1-base` decode routing trace (71,210 decode tokens, 75 MoE layers, 256 routed experts, top-8), at the live per-layer slot map `slots-7360-ctx256k.json`. Ran inside the serving image for numpy.

| policy (same 7,360 HBM slots) | mean hit | misses / layer-step |
|---|---:|---:|
| static oracle (hindsight top-S per layer, never evicts) | 0.633 | 2.94 |
| **LRU (what runs today)** | **0.719** | **2.25** |
| hybrid: pin top-50% of S + LRU tail | 0.710 | 2.32 |
| hybrid: pin top-75% of S + LRU tail | 0.697 | 2.42 |

| predictability | mean overlap | random baseline |
|---|---:|---:|
| cross-layer: L's top-8 set ∩ L+1's top-8 set, same step | **0.031** | 0.031 |
| temporal: step t set ∩ step t+1 set, same layer | **0.271** (0.07 early layers → 0.29 late) | 0.031 |

**Read:** GLM-5.3's routing has temporal locality, not global concentration, so a cache beats any fixed allocation even with hindsight (static wins only in the flat early layers 3–7, by ≤3 points). Expert ids are independent across layers: no cross-layer prefetch signal. The step-to-step signal is what LRU already exploits; a *speculative* allocator that uses the MTP draft's routing for step t+1 to issue miss fills during step t would convert the remaining ~2–4 misses/layer-step from synchronous `masked_row_copy` stalls (~18 ms/step per the E1 v2 profile) into async C2C copies. That is the version of "smarter expert placement" this trace supports.

Sim vs live: sim LRU 2.25 misses/layer-step; live K1 ~3.1, K2 ~4.4. The decode-only sim does not model prefill bypass or MTP verify positions (K2 routes 3 tokens per step). Prefetch value scales with the live number.

## Corrections logged today

- `greedy_equiv.py` executed its generation loop on `import` (the main guard was written `if __name__ != "__main__"`), silently overwriting `k2-256k-selfrepeat/greedy-r1.json` with K1 output the first time `divergence_margin.py` imported it. Caught by md5 (overwritten file == K1 r1/r2/r3 == K1-512K baseline), restored from K2 r2 (r2 == r3 byte-identical), script fixed. The `*-WRONGTEMPLATE/` dir is the first margin run, before `reasoning_effort=low` was passed to `/tokenize`; kept as a non-result.
- Earlier in the session I described the slot map as pinning specific experts. It does not: `slots-*.json` is per-layer slot *counts*; occupancy is live LRU. `EXACT_PIN` is `cudaHostAlloc` exact-size host pinning, unrelated to expert choice.

## Files

- `k1-256k-gate/`, `k2-256k-gate/`: `speed-reps.{json,log}`, `margin-on-K{1,2}.{json,log}`, `selfcheck-*.log`; K1 also `needle-ladder.{jsonl,log}`; K2 also `greedy-r1.json`, `greedy.log`
- `k1-256k-gate-WRONGTEMPLATE/`: discarded first margin run (non-result)
- `trace-analysis/`: `trace_analysis.json`, `slots-7360-ctx256k.json` (input map)
- `scripts/`: `divergence_margin.py`, `selfcheck.py`, `speed_reps.py`, `needle_ladder.py`, `trace_analysis.py`, `greedy_equiv.py` (fixed)
- `SUMMARY.json`, `throughput.csv`

Container state at close: K2 gate container up on the lab port; K1 daily stopped and kept (`glm53-big-sc13g-mtp-ctx256k-DAILY-20260913`). No restore performed.
