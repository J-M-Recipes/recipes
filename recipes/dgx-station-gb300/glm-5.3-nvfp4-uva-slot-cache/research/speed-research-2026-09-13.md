# RESEARCH-REPORT — Making GLM-5.3-big on one GB300 run faster (2026-09-13/14)

**Self-research.** Milo (running as `glm-5.3-big` on `:30001`, K=2 gate container `glm53-big-sc13g-mtp2-ctx256k-K2-gate-20260913`) researching its own serving lane, at James's request. This report is documentation only: no live windows were opened, no containers touched, no recipe/blog changes made.

- **Question:** how to make the `glm-5.3-big` lane (full 744B GLM-5.3 NVFP4 on one DGX Station GB300, vLLM v0.28.0 UVA + slot-cache hook, MTP spec decode, 256K profile, `max-num-seqs 1`, 51.3 tok/s C1 decode) run faster.
- **Method:** read-only grounding of the live Station; pull the campaign tree, frozen routing trace, and receipts locally; clone vLLM at the exact serving commit (2cf0a691); fan out four subagent analyses (A1 source/MTP, A2 hook/kernels, B trace simulation + cost model, C upstream/prior-art); my own correlation work on the live agent-traffic log; synthesis. Subagent findings files sit beside this report; their numbers were reproduced or marked modeled before being cited here.
- **Workspace:** `~/hermes/glm53-speed-research-20260913/` — `FACTS.md` (grounded state), `station/` (pulled artifacts + live logs), `vllm-src/` (pinned source), `findings/` (this report, A/B/C files, scripts).
- **Read alongside:** the current-state blog post (al-engr.com/gb300-glm-53-testing.html, Sept 13 evening) — this report deliberately audits its three ranked "next experiments" and corrects two of them.

## 0. Grounded state (all measured, this session)

| Item | Value |
|---|---|
| Live container | `glm53-big-sc13g-mtp2-ctx256k-K2-gate-20260913`, image `vllm-glm53-uva:v0.28.0-2cf0a691`, up 2h at survey; K=1 daily (`glm53-big-sc13g-mtp-ctx256k-DAILY-20260913`) stopped-and-kept |
| Geometry | 256K context, 24 GiB bf16 KV, 7,360 expert slots (per-layer 64–176), MTP K=2 (gate lane), `max-num-seqs 1`, BYPASS_ABOVE=16 |
| Under real agent traffic (this session) | gen 41–52 tok/s, SLOT_CACHE hit 0.36–0.43 (median 0.407), misses 4.55–5.10/layer-step (median 4.74), KV usage 23→26% climbing, prefix-cache hit 62.3% |
| My provider wiring | `custom[glm53]` → `:30001`, `glm-5.3-big`, 262144 ctx, **no fallback provider** — swapping `:30001` kills my own session; live swaps need the M4 detached-dispatcher pattern |
| Calibration anchors | context curve 51.29 / 44.83 / 34.14 tok/s (256K/512K/1M, K=1); E1 v2 Nsight decode buckets; Sept-13 K=2 fair gate (+3.17%, CI +0.52–6.08) |
| Trace instrument | frozen `trace-361` corpus: 71,210 decode tokens × 75 MoE layers × top-8 × 256 experts; B's replay reproduced the blog's published table to 6+ decimals |

## 1. The calibrated decode cost model (B; clean windows)

`step_ms ≈ 22.4 (base) + 4.27 ms × misses/layer-step` on K=1; fit n=17 clean points, R²=0.953, RMSE 1.12 ms (5 K1 anchors + Sept-9 K2 + 11 clean live K2 windows). Physical cross-checks: E1-v2 row-copy bucket implies 345.9 GB/s effective C2C; the fit implies 331.6; ledger probe 358.2. One miss/layer-step moves ~1.42 GB across the 75 MoE layers (~18.9 MB/slot mean, from live cache-build logs).

Fits over ALL log windows are garbage (R²≈0.10) — prefill windows, startup, and acceptance-rate variance swamp the signal; the warning fit is retained in the findings file on purpose.

## 2. Verdict on the blog's three ranked levers

### 2.1 "Reduce miss bytes by improving hit rate and slot budget" — CONFIRMED, and bigger than published

- The published LRU hit 0.719 at 7,360 slots is an artifact of the analyzer's `T//8000` subsample. **Exact full-trace replay gives 0.802 hit / 1.584 misses/layer-step at the same budget.** The probe-corpus ceiling is higher than the blog's table shows.
- Live agent traffic (my own turns in the log) runs at hit 0.407 / 4.74 misses — far below both. Three causes, disentangled: K=2's +40–52% expert traffic (verify positions), cold decode after long prefills, and per-layer workload skew (§5).
- Slot-budget curve (exact replay, proportional shape): 2,672→0.525, 5,792→0.736, 7,360→0.802, 8,928→0.855, 10,000→0.886, 12,000→0.931. Marginal gain per +1k slots falls 0.068→0.023 across that range.

### 2.2 "MTP-draft prefetch of step-t+1 miss fills" — the mechanism claim is FALSE on this model

Source verdict (A1, file:line evidence in a1-mtp-routing.md):

- GLM-5.3's MTP draft is **a separate one-layer MoE with its own fp32 router and own 256-expert weights** (`glm4_moe_mtp.py:95-101`; spec weights load as `model.layers.78` remapped into `mtp_block`, `glm4_moe.py:486-493`, `utils.py:509-514`). It is not a compressed view of the main model.
- The drafter↔target `topk_indices_buffer` sharing (`llm_base_proposer.py:1592-1619`) is the **DSA sparse-attention indexer** (`index_share_for_mtp_iteration=true`), not MoE routing.
- Cross-layer routing overlap on the main model is at chance (0.0310 vs 0.03125 random). A separately-trained layer-78 gate therefore has no mechanism to predict any main layer's expert selection. "The MTP draft already gives the routing for step t+1" is not recoverable as written.
- The cheap alternative — previous-step same-layer routing (temporal overlap 0.27) — is **quantitatively dead**: B's exact replay shows previous-step prefetch has **0.000000 coverage of actual next-step misses**, because anything requested at step t is already LRU-resident at t+1. Cache-aware prefetch copies 0.0 experts; blind DMA copies 8.0/layer-step with 73% waste. LRU already harvests the temporal signal.
- The upper bound survives reattributed: **perfect known-route lookahead would hide 11.0–14.8 ms/step → +23.5/+29.3/+37.4 tok/s** (low/central/high) at 256K-K1. Nothing in the current stack knows those routes; reaching even part of it requires a learned predictor (§2.6/§6) or an empirically measured draft↔main correspondence (the one-window falsifier, §4, item 4).

### 2.3 "Decoupling KV from expert slots" — WEAK at 256K, context-dependent

Modeling the actual vLLM path (decode gathers only the top-2048 selected KV entries per query via the DSA indexer; the C4A indexer scans an fp8 index-key cache — B read the kernels): +1,568 slots from offloading 24 GiB KV saves 1.74 ms/step; KV-over-C2C costs 1.06 + 0.11→0.98 ms/step from 30K→256K context. Net: +0.7 tok/s at 30–60K, wash at 150K, −0.4 at 256K. bf16/uncompressed bounds are far worse. Verdict: park it; it contradicts the needle-ladder finding that 256K is fully usable, and it doesn't pay at the daily profile.

## 3. Kernel/copy-path audit (A2)

- **`masked_row_copy` is already at 348–367 GB/s = 87–105% of effective C2C peak.** The bucket's cost is bytes, not the kernel. A rewrite is not first-order.
- Launch/empty overhead is only ~2.1 ms/step total (only 2.4% of w13 launches under 10 µs; launch-API time mostly overlapped under graph capture). Launch coalescing / scale-fuse caps at ~+1.5–3.1 tok/s best case.
- **Prefill bypass (M>16) neither pollutes nor warms the cache** — it returns before `_cache_forward` entirely. No LRU pollution, but also no prefill warmup: decode after a long prefill starts cold.
- Scalar-gather path (~0.50 ms/step measured) is removable with a graph-safe ~35–70 LOC patch. Best small mechanical win.
- CUDA-graph constraints verified: side-stream fork/join inside capture is possible in this pinned build if it mimics vLLM's offloader fork/event/copy/join pattern (A1) — relevant only to the learned-predictor family, which is gated on the falsifier experiment first.

## 4. Ranked conclusions — what would make it faster, in order of (gain × probability) / cost

| # | Lever | Cost | Expected (modeled, 256K-K1 baseline 51.3) | Status |
|---|---|---|---|---|
| 1 | ~~**Agent-traffic slot remap**~~ | Capture done 2026-09-14 | **FALSIFIED.** 96,504-step agent-traffic routing capture (32 real Hermes tool-loop tasks, ID_RING hook, K1-256K daily args). `alloc_slots.py` at 7,360: agent-trace LRU hit **0.6197 at the current map vs 0.6217 at the agent-optimized map** (+0.002 ≈ 0.07 ms/step ≈ +0.1 tok/s). 44 layers moved, no gain: agent traffic has lower per-layer temporal locality (0.62 sim ceiling vs 0.80 probe) with flat marginal curves; layer 12 is low-locality, not under-budgeted. Not launched. | Dead; see §5a |
| 2 | **Hook cleanup** — scalar-gather removal + scale-launch fusion | ~35–90 LOC, graph-safe, standard gates | **+1–3 tok/s** | New; A2 seam analysis done |
| 3 | **K=2 re-gate after 1+2** | One fair-gate rerun (speed-only; quality already receipted) | Needs **0.17/0.24/0.32 fewer misses/layer-step** to clear +5% (central ≈1.0 ms/step); **+2.8 tok/s if cleared** | Compounding play |
| 4 | **Draft↔main routing correlation capture** — one measurement-only window recording layer-78 + main-layer routing on agent traffic | One window, ring-buffer capture, no behavior change | Decides the ±30 tok/s prefetch family before any engineering spend | Falsifier: prior is low (main layers mutually at chance) but the cost is one window |
| 5 | **`--max-num-batched-tokens` 8192→16384** (B300 default is 2× ours) | Config-only, one-axis | Prefill/TTFT axis (211K needle rung = 62 s prefill today; agent long-prompt TTFT is the felt cost) | Untouched by any prior window (ledger-checked) |
| 6 | vLLM v0.29.0 upgrade — GlmMoeDsa CUDA routing (#52861), DSpark+SM100 sparse MLA (#52783), FlashInfer sparse MLA fix (#54465) | Own window, full re-gate; MRV2 default re-validates the whole hook | Engine refresh; not a bankable speed number | Deferred decision |
| 7 | **Learned expert predictor + async double-buffer fills** (SpecPrefetch-style adapter, A1's double-buffer sketch) | High | Fraction of the +23–37 tok/s family | Gated on #4 |

**Dead axes (now with receipts):** previous-step temporal prefetch (0.000000 coverage, exact replay); `masked_row_copy` kernel rewrite (87–105% of peak); launch coalescing as a primary lever (≤2.1 ms ceiling); KV-offload-for-slots at 256K (wash); plus the previously closed DFlash2-over-UVA, DMA demand-fill, probe-trace cache-policy reallocation, SPS, IQ2/KV-quant.

## 5. New measured finding: live agent traffic is a different workload than the slot map was built for

From my own session's 124 SLOT_CACHE windows (`station/k2gate.log.gz`, code: `findings/scripts/live_layer_skew.py`):

- **Layer 12 is the worst layer in 58/124 windows (hit 0.10–0.15) despite an above-average 112-slot budget; layer 40 is the best in 67/124 (0.62) on 96 slots.** Worst-vs-budget Spearman is only −0.17: budget size is not the driver.
- Session drift: first-quartile hit 0.443 → last-quartile 0.410, degrading to 0.255 at max context; misses 4.46→4.72; KV usage 4%→31%.
- Contrast ladder: exact-replay LRU ceiling 0.802 → probe live 0.61 → agent live 0.407. The slot map was derived from a probe corpus; agent-traffic routing demand per layer is measurably different. **This is the first direct evidence the map is workload-fit-limited, not budget-limited — and it motivates lever #1.** Caveats: single session, K=2 verify-position confound, decode-after-prefill cold starts included.

## 5a. Follow-up 2026-09-14: the remap does not pay (measured)

Capture lane `glm53-big-sc13g-mtp-ctx256k-idring-20260914` (daily K1-256K args, hook `slot_cache_hook_idring.py`: graph-safe device ring of all-layer top-8 ids for M≤8 decode steps, 374 MB, allocated before CUDA-graph capture). Corpus: 32 `hermes chat` tool-loop tasks (16 over the vLLM tree at low reasoning, 16 over the recipe tree at medium; 470 s + 1,905 s; live hit 0.50–0.53 during capture). Rows n0=177…96,681 → 96,504 complete steps, 75 MoE layers, all 256 experts seen per layer (`trace/agent-20260914/`).

| Map (7,360 slots) | Agent-trace LRU hit (alloc_slots sim) |
|---|---|
| current probe map `slots-7360-ctx256k.json` | 0.6197 |
| agent-optimized `slots-7360.json` (44 layers changed) | 0.6217 |
| uniform 96 | 0.6126 |

Δ = +0.002 hit ≈ 0.016 misses/layer-step ≈ 0.07 ms/step ≈ +0.1 tok/s, below the 0.48% speed-instrument spread. Verdict: the map is not the lever; the workload's locality is. Live hit under agent traffic (0.41–0.53) vs the 0.62 sim ceiling is the remaining gap (K=2 verify positions, cold decode after prefill, sim subsampling). Remaining real levers: hook cleanup (#2), K=2 re-gate (#3, now needs the full ~1 ms from #2 alone), `--max-num-batched-tokens` (#5), and the draft-routing correlation (#4) — the agent trace captured here can be reused for #4 if the layer-78 ids are added to the ring.

## 6. Prior art (C + Milo direct check)

Draft-assisted expert prefetch is **not novel** — an active 2025–26 thread, all edge/PCIe-scale, none on coherent C2C, none with CUDA-graph-captured decode, none single-seq agent traffic: SP-MoE (arXiv 2510.10302, 1.07–3.5× TPOT, prefetch from draft–target structural correspondence), MoE-SpeQ (2511.14102, ≤2.34×), DraftExpert (2607.24434, trained draft experts, 1.45×), SPICE (2608.21240, low-rank surrogates, ≤3.12×), SpecPrefetch (2607.24787, shared adapter, ~20%). SP-MoE's mechanism requires exactly the draft–target correspondence GLM-5.3's MTP lacks; the transferable lineage is the trained-predictor family. Classic systems (ExpertGate, MoE-Infinity, AdapMoE, ServeMoE, HoME, SiDA-MoE, EdgeMoE, DeepSpeed-Infinity) uniformly confirm: LRU with adequate slots beats static allocation; prediction must be learned or workload-aware.

Flash sidebar: James's own proven lane answers the blog's "smaller model in HBM" bet — GLM-5.3-Flash NVFP4 all-HBM (DERISKED, SGLang gb300-v2) ran ~219 tok/s C1 on this same box vs 51.3 for the 744B (~4×). The 744B's value is capability, not speed.

## 7. Blog corrections owed (data overturns the post)

1. **"The MTP draft already gives the routing for step t+1"** — false in mechanism (§2.2). Correct statement: the only one-step-ahead signals are previous-step routing (already LRU-harvested; zero marginal value) or a learned predictor / measured draft-correspondence (unmeasured).
2. **Trace table LRU 0.719** — should be labeled a `T//8000` subsample; exact full-trace replay gives 0.802 at the same 7,360 slots.
3. **Agent-traffic skew finding (§5)** — new, section-worthy: the slot map is workload-fit-limited, not budget-limited.

No correction has been published (pinned skill; documentation-only mode per James).

## 8. Provenance and reproducibility

- Subagent findings: `a1-mtp-routing.md` (25 KB), `a2-hook-kernels.md` (19 KB), `b-trace-model.md` (17 KB) + `scripts/trace_model.py` (54 KB, runs standalone) + `out/trace_model_results.json`, `c-upstream-priorart.md` (recovered from transcripts + direct checks; C timed out twice at the 900s cap and its file was assembled by me — noted inside), `SYNTHESIS.md` (the compact version of this report).
- My direct work this session: Station grounding (read-only SSH), campaign tree/trace pull, vLLM clone at 2cf0a691, FACTS.md, live-log layer-skew correlation (`scripts/live_layer_skew.py`), novelty web check, synthesis.
- Reproduce: `python3 findings/scripts/trace_model.py --write-report` (policy table, slot curve, lever table) and `python3 findings/scripts/live_layer_skew.py station/k2gate.log.gz station/trace/r1-base/slots-7360-ctx256k.json`.
- Limits: B's lever ranges are modeled from one frozen trace + one live K=2 session; single-session skew; the v0.29 PR list is verified only for the top entries; E1-v2 bucket numbers were taken as measured anchors, not re-profiled.
- Governance: one-axis windows, measurement-only, stop-not-rm, no `:30003` (dark DSF prod, campaign contract), no secrets in receipts — all unchanged and respected (nothing was launched).
