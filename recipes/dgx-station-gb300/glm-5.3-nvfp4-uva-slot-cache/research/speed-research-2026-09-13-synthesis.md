# GLM-5.3-big GB300 serving-stack speed research — synthesis (2026-09-13/14)

Self-research: Milo (glm-5.3-big via :30001, K=2 gate container) researching its own serving lane, per James's ask: "I'd like it to run faster." Sources: live Station state (read-only), frozen campaign artifacts pulled to `station/`, vLLM v0.28.0 source at the exact serving build, four subagent analyses (A1 source/MTP, A2 hook/kernels, B trace/cost model, C upstream/prior-art), plus my own live-log correlation work. All findings files in this directory; every number below is tagged measured/modeled/simulated.

## Verdict on the blog's three ranked levers

### Lever 1 — "Miss-byte reduction via hit rate and slot budget": CONFIRMED, sharpened
- Calibrated cost model (clean windows, R²=0.953): **4.27 ms/step per miss/layer-step** (E1-derived central 4.09; effective C2C 332–346 GB/s vs ledger probe 358). One miss/layer-step moves ~1.42 GB across 75 layers.
- The blog's LRU 0.719 at 7,360 slots is a **T//8000 subsample artifact**: exact full-trace replay gives **0.802 / 1.584 misses** at the same budget. The probe-corpus ceiling is higher than the published table; the live-vs-ceiling gap is bigger.
- **Live agent traffic (this session, K=2) runs hit 0.407 / 4.74 misses/layer-step** — vs 0.61/3.12 on probes and 0.802 exact-replay ceiling. Worst-layer analysis: **layer 12 is worst in 58/124 windows (hit 0.10–0.15) despite an above-average 112-slot budget; layer 40 is best in 67/124 (0.62) on 96 slots.** Budget size is weakly anti-correlated with badness (Spearman −0.17) → the probe-derived per-layer slot map mismatches real agent-workload routing. A2 confirmed prefill bypass neither pollutes nor warms the LRU, so the gap is K=2 expert-traffic (+40–52%) + cold decode after long prefills + per-layer skew.
- **New actionable: regenerate the slot map from an agent-traffic routing capture** (same 7,360 total, take from layer-40-likes, give to layer-12-likes). Config-only (`alloc_slots.py` + new `SLOT_CACHE_PER_LAYER` JSON), graph-safe (per-layer shapes fixed at init), zero kernel risk. This is NOT the closed "equal-HBM reallocation" axis — that replayed the frozen probe trace; the new fact is that agent demand per layer is measurably different.

### Lever 2 — "MTP-draft prefetch of step-t+1 miss fills": MECHANISM CLAIM IS FALSE as stated
- **Source verdict (A1, file:line evidence):** GLM-5.3's MTP draft is a separate one-layer MoE (`glm4_moe_mtp.py:95-101` constructs its own `Glm4MoeDecoderLayer` with own fp32 gate + 256 experts; spec weights load as `model.layers.78` remapped into `mtp_block`, `glm4_moe.py:486-493`, `utils.py:509-514`). The drafter↔target `topk_indices_buffer` sharing (`llm_base_proposer.py:1592-1619`) is for the **DSA sparse-attention indexer** (`index_share_for_mtp_iteration`), not MoE routing.
- Draft-layer routing therefore cannot predict main-layer routing: all 75 main layers are mutually at chance (cross-layer overlap 0.031 vs 0.03125 random), and a separately-trained layer-78 gate has no mechanism to beat that.
- **Quantitative kill of the cheap variant (B, exact replay):** previous-step same-layer prefetch has **0.000000 miss coverage** — anything requested at step t is already LRU-resident at t+1. Cache-aware prefetch copies 0.0 experts; blind DMA copies 8.0/layer-step with 73% pure waste. The 0.27 temporal overlap is real but LRU already harvests it.
- **The upper bound survives, reattributed:** perfect known-route lookahead would hide 11.0–14.8 ms/step → **+23.5/+29.3/+37.4 tok/s** (low/central/high) at 256K-K1. Nothing in the current stack knows those routes. The literature path to partial coverage is a **trained predictor** (SpecPrefetch-style adapter / SP-MoE draft-target correspondence / DraftExpert draft experts) — see prior art below.
- **Cheap falsification experiment before building anything:** one capture window recording NextN(layer-78) routing + main-layer routing simultaneously on agent traffic (ring-buffer extension, capture-only, no behavior change). Prior is low (main layers mutually at chance), but the cost is one window and it green-lights or kills the whole prefetch family honestly.

### Lever 3 — "KV↔slot decoupling": WEAK at 256K, context-dependent
- B modeled the actual vLLM fp8-indexer path (sparse decode gathers only top-2048 selected entries; C4A indexer scans an fp8 index-key cache): +1,568 slots buys 1.74 ms/step; KV-over-C2C costs 1.06 + 0.11–0.98 ms (30K→256K). Net: +0.7 tok/s at 30–60K, wash at 150K, −0.4 at 256K. bf16/uncompressed bounds are much worse.
- Verdict: not a top lever for the 256K daily profile; potentially a small win only if the daily context were shrunk (contradicts the needle-ladder finding that 256K is fully usable — so: park it).

## What else is real (ranked, with costs)

1. **Agent-traffic slot remap** (config-only, one capture window + regen + one launch). Expected live-hit 0.41 → 0.55–0.65 territory if skew dominates; Δ+1–4 tok/s on agent traffic. Honest caveat: probe-corpus speed likely unchanged; this is an agent-experience lever, and it must be validated on agent-like traffic, not speed_reps alone.
2. **Hook mechanical cleanup** (A2: graph-safe, 35–90 LOC total): scalar-gather removal (~0.50 ms/step measured bucket) + scale-tensor launch fusion (slice of ~2.1 ms launch/empty overhead). Expected +1–3 tok/s combined. Gates: greedy self-repeat 20/20 + speed reps.
3. **K=2 re-gate after 1+2** — the compounding play. B: K=2 needs only **0.17/0.24/0.32 fewer misses/layer-step** (low/central/high; ≈1.0 ms/step central) for 52.92 tok/s to clear the +5% bar (54.08). If the remap + cleanup deliver ≥1 ms/step on the K2 lane, rerun the Sept-13 fair gate. K=2 already has quality-equivalence receipts (logprob parity 0.0, ladder 191/200, 0 losses), so a speed-only re-gate is cheap.
4. **Draft-routing correlation capture** (measurement-only, one window): decides the prefetch family's fate before any engineering spend.
5. **`--max-num-batched-tokens 16384`** (B300 default is 2× our 8192): prefill/TTFT axis, config-only, one-axis. The needle ladder's 211K-token rung took 62 s prefill; agent long-prompt TTFT is the felt cost of this lane. Untouched by any prior window (A2 checked the ledger).
6. **vLLM v0.29.0 upgrade** (own window, full re-gate): GlmMoeDsa CUDA routing (PR #52861), adaptive DSpark verification with SM100 sparse MLA validated on GLM-5.2-NVFP4 4×B300 (PR #52783), FlashInfer sparse MLA BLHNC fix (PR #54465). Risks: MRV2 default changes execution paths; the Triton slot-cache hook needs revalidation against MRV2 + graph capture + non-compiled routing. Upgrade is an engine-refresh axis, not a speed lever to bank on.
7. **Learned expert predictor + async double-buffer fills** (the only path toward the +23–37 tok/s family): adapter trained on captured routing (SpecPrefetch-style), A1's double-buffer/pending-bookkeeping design sketch, second-stream-in-capture following vLLM's offloader fork/join pattern. High effort; gate on experiment 4's result first.

## Dead axes (keep closed, now with receipts)
- Previous-step temporal prefetch: 0.000000 coverage (B, exact replay).
- `masked_row_copy` kernel rewrite: already at **348–367 GB/s = 87–105% of effective C2C peak** (A2). Bytes are the cost, not the kernel.
- Launch coalescing as a primary lever: capped at ~2.1 ms/step overhead; ≤+3 tok/s best case.
- KV-offload-for-slots at 256K: wash (B, both fp8-indexer and bf16 bounds).
- DFlash2-over-UVA, DMA demand-fill, probe-trace cache-policy reallocation, SPS, IQ2/KV-quant: previously closed, unchanged.

## Prior art (C + my novelty check)
- Draft-assisted expert prefetch is **not novel**: MoE-SpeQ (arXiv 2511.14102, up to 2.34× on PCIe offload), **SP-MoE (arXiv 2510.10302, 1.07–3.5× TPOT, "speculative expert prefetching that exploits structural correspondence between draft and target models" + cutoff-layer policy + pipelined async runtime)**, DraftExpert (2607.24434, trained draft experts, 1.45×), SPICE (2608.21240, low-rank surrogates, 3.12×), SpecPrefetch (2607.24787, shared adapter, ~20% edge). All edge/PCIe-scale; **none on coherent C2C/Grace-Blackwell, none with CUDA-graph-captured decode, none single-seq agent workloads.**
- Critical transferability note: SP-MoE's mechanism requires draft–target structural correspondence. GLM-5.3's MTP lacks it (own experts, own router). The transferable variant for us is the **trained predictor** lineage (SpecPrefetch/DraftExpert) or measuring layer-78 correspondence empirically (experiment 4).
- Older generation (ExpertGate input-embedding prefetch, MoE-Infinity activation-aware, AdapMoE, ServeMoE, HoME, SiDA-MoE, EdgeMoE, DeepSpeed-Infinity) all confirm the same principle our trace already proved: LRU-with-enough-slots is hard to beat by allocation; prediction must be learned, not structural.
- Flash sidebar: James's own proven lane already answers the blog's bet — GLM-5.3-Flash NVFP4 all-HBM on this box (DERISKED, SGLang gb300-v2) ran ~219 tok/s C1 vs the big model's 51.3. All-HBM small-model wins on speed by ~4×; the 744B's value is capability. (catid/dgx_station_benchmarks has a glm-5.3-flash section; NVIDIA forum thread 382044 discusses DFlash2 on GB300 Station.)

## Blog corrections owed (data overturns the post — James's rule)
1. Sept 13 evening section: "issuing next-step miss fills asynchronously from the MTP draft's routing" — the MTP draft does not know main-layer routing (source-verified). The correct statement: the only one-step-ahead signals are previous-step routing (already LRU-harvested, zero marginal value) or a learned predictor / measured draft-correspondence (unmeasured).
2. Trace table: LRU 0.719 at 7,360 slots should note it is a T//8000 subsample; exact full-trace replay gives 0.802. Ceiling higher than published.
3. New workload finding worth a section: agent traffic (hit 0.407, layer-12-class skew) differs materially from the probe corpus the slot map was derived from — the first direct evidence that the map is workload-fit-limited, not budget-limited.

## Reproducibility
- `findings/a1-mtp-routing.md`, `a2-hook-kernels.md`, `b-trace-model.md` (+ `scripts/trace_model.py`, `out/trace_model_results.json`), `c-upstream-priorart.md` (partial, recovered via transcript + this synthesis).
- Live correlation (layer worst/best vs budget) computed from `station/k2gate.log.gz` + `slots-7360-ctx256k.json` (this session, code inline in transcript).
- FACTS.md holds the grounded state; station/ holds the pulled campaign tree.
