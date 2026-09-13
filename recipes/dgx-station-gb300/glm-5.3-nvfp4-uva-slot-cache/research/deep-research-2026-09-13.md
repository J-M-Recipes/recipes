# Big GLM-5.3 on one GB300 — deep research pass

**Date:** 2026-09-13 · **Author:** Milo (session model dsv41-flash-uva)
**Scope:** external ecosystem state + our lane's measured ground truth + a ranked improvement menu.
**Station:** read-only checks only; nothing modified. The box is currently mid-DSV4-campaign (`dsv41-vllm-v12-1M-k5-off60-util97-BOUND-REF` on :30006, up 12 h); all big-GLM containers are stopped-kept.

---

## TL;DR

1. **Upstream vLLM is now building the same machine we built.** PR #56177 (2026-09-10) "shared GPU expert pool with a device-side planner" is architecturally our slot cache: pinned host master, GPU bank shared across layers, device-side LRU, fixed-grid row copies, one MoE call with row remap, decode stays inside CUDA graphs. Still Marlin-only, still open — but it's the clearest validation of the lane, and a place to align and share our data.
2. **A new upstream bug class is a candidate explanation for our K2 mystery.** #54945: FlashInfer CUTLASS NVFP4 MoE's *fused finalize* is not bit-stable for identical requests (atomics in the weighted combine); `use_fused_finalize=False` is bit-stable. Sibling #53257 (DSV4-Flash temp-0 nondeterminism). Note: that bug is a *different* kernel family than ours (we run the TRT-LLM blockscale MoE via FlashInfer), but it establishes nondeterministic MoE reducers as a real, named failure class — and makes a K2 self-repeatability probe the cheap decisive test before "byte-exact" can be treated as a fair quality gate.
3. **Our two remaining physical levers are unchanged, but one of them now has a cheap concrete instantiation.** Policy/prefetch/policy-sim axes are closed with evidence. What's left: (a) *fewer miss bytes via capacity* — trade GPU KV for expert slots using the boot-tested native KV offload; (b) *TTFT/prefill* — the bypass path costs ~0.6 s/short prompt and has never had its own experiment.
4. **Highest-value next window: "capacity trade" (E2 revisit).** Free ~24 GiB of GPU KV → ~1,214 more expert rows (~16 slots/layer on top of 48–96). Needs a small local patch (the v0.28 admission check counts GPU KV only — still true on main as of today), then a restore-protected window to validate spill + speed + quality. Prior (not claim): +5–8% C1 and -10–16% misses.
5. **Offline work that can start while DSV4 owns the box:** the v0.29/MRV2 rebase feasibility spike — v0.29.0 (2026-09-09) ships GLM-family kernel work, spec-decode metrics, determinism infra, and KV-offload fixes that touch three of our local patches.

---

## 1. External findings (since our Sept 7–9 window)

### 1.1 vLLM v0.29.0 — released 2026-09-09

Source: https://github.com/vllm-project/vllm/releases/tag/v0.29.0

| Change | Relevance to our lane |
|---|---|
| **Model Runner V2 now default for all models** (#53183); MRV1 deprecated, removal targeted for v0.32 | Our pinned build is v0.28/MRV1 with deep hooks (slot cache, FFI router, exact-pin, T1 instrumentation). Any rebase must re-validate all of it; MRV2 gap list (SP, DBO, elastic EP, some spec methods → MRV1 fallback) is the first thing to check for us |
| Spec decoding: per-request acceptance stats in the OpenAI API (#48915); adaptive verification with logprobs (#52242); SM100 sparse MLA for GLM-5.2 (#52783); DSv4+SM90 (#52795) | #48915 would replace our log-scrape acceptance instrumentation; GLM-5.2 sparse MLA is the same DSA family as our GLM-5.3 — candidate kernel benefit on rebase |
| Kernels: **masked MHA prefill for GLM-5 head dimensions** (#53785); **GLM-5.2 sparse MLA Q-concat fused** (#53878); FA4 re-enabled for head_size=256 on Blackwell (#52980) | GLM-family prefill/attention kernels — directly relevant to the TTFT/prefill axis, but only via a rebase |
| NVIDIA: DeepSeek V3.2 / **GLM-5.2 DSA routed to the optimized CUDA path on all GPUs** (#52861); cuBLAS out_dtype router GEMM incl GB10 (#54048) | Same-arch family as us; rebase-relevant |
| KV offloading: EC connector CUDA events (#49994), P2P tier request-level offload (#52912), `/dev/shm` leak fix (#52596) | Our native KV offload lane (E2) — the leak fix and event-driven connector matter if we push that path |
| Determinism: `trace_decode_token_ids` deterministic decode replay (#46701); per-arch tuned batch-invariant matmul configs (#53247); Blackwell autotuning 33.6% E2E | The infrastructure that could make a K>1 gate tractable |
| `--cpu-offload-params` now reaches vision/audio towers (#53120) | Minor for us (text-only) |

**KV-offload accounting status (checked on main today):** `check_enough_kv_cache_memory` in `vllm/v1/core/kv_cache_utils.py` still computes needed memory from KV specs and compares against GPU-provided `available_memory` — **the `kv_offloading_size` buffer is not part of the admission check**. Our E2b blocker is still present upstream as of 2026-09-13. No fixing PR found in the Sept 1–13 sweep.

### 1.2 New upstream PRs/issues directly on our lane

| # | Item | State | Impact on us |
|---|---|---|---|
| **#56177** | `--moe-expert-pool-rows`: shared GPU expert pool, device-side planner, NVFP4 Marlin experts; "no host-side routing readback... stays inside CUDA graphs (FULL_DECODE_ONLY)"; measured 7.4 → 63.2 tok/s on Qwen3.8-Flash-Next NVFP4 at a 48 GiB GPU budget | Open, merge conflicts, needs rebase | **The upstream twin of our slot cache.** Marlin-only today (we run TRT-LLM-GEN kernels), but their planner/copy design and the "frozen placement through capture" idea are worth reading line-by-line. Community thread (#56175) is the right place to post our 744B/Grace data |
| #56175 | The issuing discussion ("Qwen3.8-Flash-Next NVFP4 on a 48 GB GPU") | Open | Same as above |
| #56118 | `VLLM_EXPERTS_LOAD_DEVICE=cpu` — GPU/CPU mixed expert placement, construct-time OOM fix | Open | A cleaner "experts on host" switch than `--cpu-offload-gb`; watch, not adopt yet |
| #56410 / #56413 | Bug: `--cpu-offload-gb` uses ~1.9× host memory it asks for (pin-before-UVA duplicate); fix PR "Avoid duplicate host allocation for UVA weights" | Open | **Exactly the class our exact_pin patch fixes locally** (we measured pow2 rounding waste; upstream found the pin-before-UVA duplicate). Adopt/retire our patch when it merges |
| #54972 | UVA offloader: release the accelerator blocks it frees (fixes load OOM "no matter what N") | Open | Same family as our sc1–sc7 OOM chase; check against ours on rebase |
| #56319 | Warn when NoopOffloader is silently selected | Open | Hygiene; our lane pins offload explicitly |
| #56283 | `--cpu-offload-gb` accepted then NoopOffloader — was 0.27.1, fixed on main | Closed-ish | Not our build |
| #37190 | LFRU expert cache (`--moe-expert-cache-size`) | Still open, last updated 2026-08-28 | Still the lineage anchor; #56177 is the newer sibling |
| #38256 | Incremental MoE Expert Offloading RFC | Active — 17 comments, latest 2026-09-12 incl. our own DSV4-Flash Grace-Blackwell data (two read paths 340 vs 90 GB/s; adaptive-verification k-sweep) | The community thread where our data lives; keep posting big-GLM numbers |
| #55620 | EAGLE-3/DFlash aux hidden states for Glm5Next | Closed (landed) 2026-09-07 | **Flash model only**; big GLM remains MTP-only |
| #54976 | B12X sparse MLA + DSA backends | Open | SM120/121 (Spark) only; not our SM103 |

### 1.3 The nondeterminism thread — new hypothesis for our K2 failure

- **#54945 (2026-09-02, open):** FlashInfer CUTLASS NVFP4 MoE `use_fused_finalize=True` (the vLLM default — never passed the arg) reduces top-k expert outputs with atomics → *identical* requests at temp 0 return different logprobs; token-level divergence. `use_fused_finalize=False` = bit-stable (reproduced on two systems, GB10 aarch64 + RTX PRO 6000 sm_120). Their word: "the per-token weighted combine of expert outputs is one of the few places where a fixed-shape, identical-input call can still sum in a different order."
- **#53257 (2026-08-21):** DeepSeek-V4-Flash temp-0 nondeterminism, rate scales with concurrency; other reporters connect it to the same fused-combine step.
- **Why this matters to us:** our T2 memo left K2 divergence ambiguous — some sites near-ties (margin 0.0), some confident (margin 6.0). If the verify-path combine (or any MoE reduce) is order-nondeterministic under different batch shapes, then (a) the 9/20 was never a fair "quality" signal, and (b) the fix is *determinism*, not a looser gate. Our own kernels (TRT-LLM NVFP4 blockscale MoE) passed same-input repeatability on synthetic layers (`maxabs_diff 0.0` over 300 churn steps) — but that was the slot-cache kernel on fixed shapes, not the full production K2 verify path. **Cheap decisive test: K2 self-repeatability** (same greedy prompt ×3 on the K2 lane; if K2 ≠ K2, it's nondeterminism). We never ran that.
- **Counterpoint:** K2 also failed the speed gate (+3% vs +5% needed) with misses +44%. Even a quality fix doesn't make K2 a win today — it becomes interesting again only *after* miss bytes drop (then its amortization compounds).

### 1.4 SGLang state

- Official GLM-5.3 cookbook (docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3): supports big GLM-5.3 fp8/bf16/nvfp4; hardware list includes gb300 — **but every gb300+nvfp4 cell is multi-GPU** (tp4 + EAGLE; TTFT 307 ms, TPOT 1.71 ms, 1,113 tok/s/GPU at C1 on 4 GPUs). Single-GPU + offload remains *our* lane; SGLang has no official single-GPU big-GLM recipe.
- The official NVFP4 checkpoint for SGLang is **RadixArk/GLM-5.3-NVFP4** (see §1.5).
- No mainline "expert keep/offload" selective expert cache: code search for `expert_keep`, `KEEP_OFFLOAD`, `expert offload` → 0 hits in sgl-project/sglang. Elastic-EP host-DRAM backup (#17374, merged Feb 2026) is fault-recovery, not serving speed. The EPLB tooling (expert-distribution recorder, location updater) exists for *load balancing*, and our own profiling needs are better served by our trace hooks.
- LMSYS GB200 NVL72 Part II blog (offload expert weights to Grace + prefetch): real datacenter pattern, wrong scale for us (their link budget ≠ our 360 GB/s measured C2C), but validates host-tier experts + prefetch as architecture, not hack.

### 1.5 Checkpoints & drafters (big GLM-5.3) — verified on HF today

| Repo | Status | Read |
|---|---|---|
| incoai/GLM-5.3-NVFP4 @ 54e52520 | Our current checkpoint; no newer revision (lastModified 2026-08-28) | Unchanged |
| **RadixArk/GLM-5.3-NVFP4** | 432.9 GiB total (byte-identical class to incoai, both ModelOpt experts-only NVFP4 W4A4); released 2026-08-28; explicitly validated on B300; the NVFP4 model named in SGLang's official cookbook | A provenance cross-check / alternate if we ever need one. Not a speed lever |
| RedHatAI/GLM-5.3-MXFP4 | New (MXFP4 with published recovery evals vs FP8) | Different quant family; no case to switch |
| PhalaCloud W4AFP8 · amd Quark-MXFP4-AttnFP8 · OneNexus MXFP4 · gpustack W4A8 | New 4-bit-ish variants incl. FP8 attention variants | AttnFP8 could trim a few GiB of non-expert HBM → slightly more slots; but breaks bit-exact quality baseline → needs the full quality discussion. Low priority |
| Drafters | incoai/GLM-5.3-DFlash2 unchanged (Aug 28); all new DFlash2 activity is for the **Flash** model (Solstice-AI etc.) | No new big-GLM drafter. DFlash2-over-UVA remains closed (1.55 accepted vs 3.0 gate) |
| zai-org/GLM-5.3 official | Weights live (updated Sept 4); 141-shard ~1.5 TB BF16 base | Nothing new |

### 1.6 Alternatives (FreeToken and friends)

- **FreeToken:** does *not* support big GLM-5.3; roadmap lists DeepSeek-V4.1-Flash pending; GLM-5.3-Flash landed (#332). Their architecture — host-RAM experts + GPU LRU expert cache — is our design pattern on smaller boxes. `Cerynitius/freetoken-ox-boost` shows the same pattern winning: GLM-5.3-**Flash** NVFP4 17.8 → 36.7 tok/s single-stream on one RTX PRO 6000 via resident layers + prefetch (~84% cache hits). Good supporting evidence for the pattern; nothing to port for 744B.
- **llama.cpp #27861** (GPU LRU expert cache): similar idea, wrong stack for GLM-5.3 (DSA + NVFP4 + our patches).
- **Public single-box big-GLM numbers:** nothing new found for 744B NVFP4 single-GPU/offload since early Sept. Our 43–47 tok/s C1 lane appears to remain the strongest public single-Station result; the cookbook's multi-GPU numbers are context, not competition.

---

## 2. Our lane — measured ground truth (recap, Sept 7–10)

- **K1 incumbent (kept, stopped):** 45.75 tok/s median 512-tok; hit rate 73.5%; misses 4.236/layer-step; decode = 40.1 ms/step wall, of which `masked_row_copy` 20.1 ms (≈90% C2C bytes at ~360 GB/s ceiling), dense GEMM 9.4, routed MoE 5.4, bookkeeping 1.05, other 7.2.
- **K2:** 47.13 (+3% vs +5% gate), 9/20 greedy, misses 6.02, accept 2.39/3. **Closed pending a determinism finding.**
- **Offline sim:** calibrated to 0.81% on live K1 counters; best policy/allocation candidate cut 0.083% — policy axis closed. Prefetch precision 8.4% — prefetch axis closed (before Stage B).
- **T0** real-systemd restore-contract harness: CONTRACT_OK (PR #5). **T4** NVTX correlation: 634/634 spec_verify ranges projected.
- **E2a (KV offload, native):** boots with `--kv-offloading-size 48 --kv-offloading-backend native` + shm 64g; idle decode 45.724 vs 45.75 baseline — **idle connector cost ≈ 0**. Container `glm53-big-sc13g-mtp-ctx512k-kvoff-48g` stopped-kept on box.
- **E2b (shrink GPU KV to 24 GiB at declared 512K):** NO_GO — v0.28 admission check counts GPU KV only ("Estimated max len at 24 GiB is 274368").
- **E1 DFlash2-no-prefix re-run:** STOP (1.55 accepted vs 3.0 gate) — prefix cache was not the cause.
- **Prefill note (from E1 v2):** bypass path ≈ 7.8 ms/layer across 75 cached layers ≈ 0.6 s per short prompt — flagged for its own experiment, never done.

---

## 3. Ranked improvement menu

### Tier 1 — do next

**1. Capacity trade: hybrid KV offload → more expert slots ("E2 revisit").**
The only untested instantiation of the blog's #1 lever (fewer miss bytes). Trade ~24 GiB of GPU KV for ~1,214 expert rows (≈16 slots/layer on top of 48–96; 5,792 → ~7,000 total ≈ 0.68 planning hit vs 0.6166).
- *Prep (offline, can start now):* small local patch adding `kv_offloading_size` to the admission check (same patch discipline as our other hooks: sitecustomize, reviewed, hash-pinned); unit-test the check math; pre-write the window contract.
- *Window (restore-protected):* launch with `--kv-cache-memory 24GiB` + `--kv-offloading-size 48` + declared 512K; gates: greedy 20/20, 512K needle, C1 speed ≥ +5% vs same-window incumbent, no long-context regression cliff, hit-rate counter delta, restore proof.
- *Fallback variants:* 32 GiB GPU KV (+8 GiB → +~8 slots/layer) if spill costs bite.
- *Risks:* KV spill performance at long context (DSA sparsity makes it plausible, unproven); host memory packing (E2a already proved the 51.5 GiB buffer fits alongside pinned experts); admission-check patch is a behavior change and must stay local until upstream does it.
- *Prior (±):* +5–8% C1, misses −10–16% — planning-curve extrapolation, stated as prior, not claim.

**2. Prefill / TTFT experiment (new axis).**
~0.6 s of every short-prompt TTFT is the bypass path; agent UX cares. First step is measurement: dedicated prefill-only nsys capture on the current lane (bucket the bypass cost by layer/phase, separate prompt classes), then options — e.g., slot-fill policy during chunked prefill, chunk sizing, or (on rebase) the new GLM-family prefill kernels. Cheap to scope; medium confidence; genuinely fresh.

**3. v0.29 / MRV2 rebase feasibility spike (offline, box-free until build test).**
v0.29 brings GLM-family kernels, spec metrics, determinism infra, KV-offload fixes, and three upstream fixes in the exact bug classes our local patches cover (#56413 exact-pin analog, #54972 UVA release, #56319 guard). Deliverable: a memo — what our hooks require, MRV2 gap check, autotune re-key plan, patch retirement opportunities, estimated window cost. No live changes.

### Tier 2 — worthwhile, smaller

**4. K2 determinism closure probe.** One arm on the stopped K2 container: same prompt ×3 at K2 → bit-stable or not. Also grep our MoE path for fused-finalize-class reducers. Only *after* capacity lands does reopening K2-for-speed make sense (amortization compounds with fewer misses). Note: this is also a decision input for James's T2 gate options (keep byte-exact vs task-level gate).

**5. Fine-grained "other"-bucket attribution.** `other` is 7.2 ms/step (16.6%) and loosely explained. One short nsys window with named phase NVTX could expose 1–3 ms of reducible overhead (or confirm it's all essential small kernels). Medium-low value, low cost — fold into any window we're already paying for.

**6. Structured-output hardening (quality track).** The secondary gate (structured/supplied-data) is unresolved across *all* lanes (33–36/44), not MTP-specific; schema-constrained V2 fixed JSON validity 40/40. Worth a bounded investigation of the failure classes (parser boundary vs model) if agent tooling suffers — currently tool-chain episodes pass 4/4 everywhere, so this is a slow-burn item.

### Tier 3 — engagement / watch

**7.** Post our big-GLM capacity + K1/K2 findings to the #56175/#56177 threads (same problem, our scale is unique); watch #38256. **8.** Keep RadixArk NVFP4 as a documented alternate checkpoint. **9.** Watch #56177's planner design for anything portable to our TRT-LLM path.

---

## 4. Closed axes — do not revisit (with evidence)

- Cache policy/hit-rate optimization (sim calibrated; best +0.083%), one-layer-ahead prefetch (8.4% precision), copy-engine vs zero-copy (equal at ceiling), DMA demand-fill (slower), MoE GEMM tuning (small), CUDA-graph enablement (already captured), empty-mask launch coalescing (≤2 ms, mostly bytes not launches), DFlash2 K4 (accepted 1.55/3.0), MTP K2 for speed (+3%/+5% gate), sub-4-bit quants (policy), KV quant (policy).

## 5. Process note

The initial 4-stream fan-out lost its parent hook mid-run (session hiccup); two streams had finished their gathering, one completed fully, all raw artifacts were salvaged from `~/hermes/research/big-glm-2026-09-13/` (raw/, github/, patches/, source_*.txt, vllm-new-items.json). I verified every load-bearing claim myself against primary sources (vLLM main, PR pages, HF API, SGLang cookbook files) before writing this report.

## 6. Key sources

- vLLM v0.29.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.29.0
- #56177 expert pool: https://github.com/vllm-project/vllm/pull/56177 · #56175: https://github.com/vllm-project/vllm/issues/56175
- #54945 NVFP4 fused-finalize nondeterminism: https://github.com/vllm-project/vllm/issues/54945 · #53257: https://github.com/vllm-project/vllm/issues/53257
- #56410 / #56413 host-memory duplicate: https://github.com/vllm-project/vllm/issues/56410
- #54972 UVA release fix: https://github.com/vllm-project/vllm/pull/54972
- #37190: https://github.com/vllm-project/vllm/pull/37190 · #38256: https://github.com/vllm-project/vllm/issues/38256
- KV admission check on main: `vllm/v1/core/kv_cache_utils.py` (fetched 2026-09-13)
- SGLang GLM-5.3 cookbook: https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3 · v0.5.19 notes: https://github.com/sgl-project/sglang/releases
- RadixArk checkpoint: https://huggingface.co/RadixArk/GLM-5.3-NVFP4
- FreeToken: https://github.com/FlashML-org/FreeToken · ox-boost: https://github.com/Cerynitius/freetoken-ox-boost
- Our blog/recipe: https://al-engr.com/gb300-glm-53-testing.html · https://github.com/J-M-Recipes/recipes/tree/main/recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache
