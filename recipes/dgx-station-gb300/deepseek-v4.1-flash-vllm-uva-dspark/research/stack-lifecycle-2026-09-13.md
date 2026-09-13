# Current research round — DSV4.1 serving (2026-09-13 ~06:40 CDT)

Scope: what has changed in the world since the v12 campaign closed 2026-09-12 18:30.
Everything below verified from live sources this morning; URLs/PR numbers inline.

## 1. HEADLINE: the branch our serving image came from no longer exists

Our lane is pinned to `vllm/vllm-openai:deepseekv41-flash-0909`, built from fork branch
`dsv41-feat` @ `e47aa78` (**Sep-4 base, 377 commits behind main** — this was a standing
risk we documented in the neighbor lessons and never resolved).

- `dsv41-feat` → **HTTP 404. Deleted.** (verified: `gh api repos/vllm-project/vllm/commits?sha=dsv41-feat` → Not Found)
- **PR #56228 "[Model] DeepSeek-V4.1-Flash Model Definitions" MERGED to main 2026-09-10T12:16:26Z**
  (47 files, +12,902/−196). Now `vllm/models/deepseek_v4_1/` and `vllm/models/deepseek_v4/` on main.
- PR #56512 body confirms the mechanism: *"The original PR was stacked on the `dsv41-feat`
  staging branch and was auto-closed when that branch was removed after DeepSeek V4.1 merged to main."*
- Docker tag `deepseekv41-flash-0909`, built 2026-09-10T06:47 — **~6 hours BEFORE the merge**.
  It is the last DSV4.1 dedicated tag and it predates the merge, so it does not contain it.
- Newest images: `vllm/vllm-openai:nightly` @ 2026-09-13T06:15, `cu129-nightly` @ 06:27.
- Latest vLLM release: **v0.29.0, 2026-09-09** (predates the merge).

**Consequence:** our recipe's own stack is now a dead-end fork against a deleted branch.
The rebase target exists. Feasibility detail in §6 (subagent report pending).

## 2. What mainline kept — our recipe's dependencies survive (checked in main tree)

| Dependency | Main status | Path |
|---|---|---|
| UVA expert offload | **present, same shape** | `vllm/model_executor/offloader/uva.py`, class `UVAOffloader`, `cpu_offload_gb` / `cpu_offload_params` |
| Offload CLI config | present | `vllm/config/offload.py` — `UVAOffloadConfig`, `PrefetchOffloadConfig` |
| Engram CPU offload | **present, and now DEFAULT ON** | `vllm/config/engram.py` — `cpu_offload` "Defaults to `VLLM_PLE_CPU_OFFLOAD`, which is enabled by default" |
| Engram impl | present | `vllm/models/deepseek_v4_1/nvidia/engram.py` |
| DSpark spec decode for V4.1 | present | `vllm/models/deepseek_v4_1/nvidia/dspark.py`, `vllm/v1/worker/gpu/spec_decode/dspark/` |
| MXFP4 trtllm MoE kernel | **still registered** | `trtllm_mxfp4_moe` in `docs/design/moe_kernel_features.md` (`TrtLlmMxfp4ExpertsMonolithic` / `Modular`), quant via `vllm/model_executor/layers/quantization/mxfp4.py` + `fused_moe/oracle/mxfp4.py` |
| `--long-prefill-token-threshold` | present | `vllm/engine/arg_utils.py` |
| nvfp4_ds_mla (FlashMLA #221) | **not on main yet** | only our fork image; main has `csrc/.../nvfp4_ds_mla_cache*` but the attention wiring is PR #56344, OPEN |

Notable: the deepseek_v4_1 kernels moved out of `model_executor/layers` into a new
`vllm/models/deepseek_v4_1/` tree (model definitions + per-vendor kernels). Also
`vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py` is not on main as a
file — the trtllm mxfp4 experts appear in the kernel table, so verify the exact path
before rebasing (subagent report covers this).

## 3. The offload-development front — two new PRs that did not exist in our campaign

**PR #54610 — `[Bugfix][Offloader] Prioritize sparse MoE experts in UVA CPU offloading`**
(open, created 2026-08-31, updated 2026-09-10, 3 files, +539/−62, fixes issue #54593)
This is a **direct, independent confirmation of our `PARAMS` workaround** — and it fixes
the bug for everyone who doesn't pass it. Issue #54593:

> "`--cpu-offload-gb` evicts in declaration order, which for MoE offloads the hottest weights first"

Measured there on Qwen3.8-Flash-Next, 1 card, `--cpu-offload-gb 47`:
**default order 4.26 tok/s vs `--cpu-offload-params experts` 9.46 tok/s = 2.2×, one flag.**
That is the same failure mode we route around with
`--cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight`.
The PR makes that the default when the model is MoE. **Not merged.**

Relevance to us: our recipe hardcodes `PARAMS`, so we are already correct — but a
mainline rebase would let us drop the flag, and more importantly this is the second
independent measurement of the same trap. Worth citing on the blog and in the RFC thread.

**PR #56118 — `[MoE] add VLLM_EXPERTS_LOAD_DEVICE=cpu for GPU/CPU mixed expert placement`**
(open, created 2026-09-09, 3 files, +59/−2)
Lets routed-expert weights be **constructed on the host** instead of GPU, so a MoE whose
experts don't fit HBM is constructible at all. Motivation is a 40 GB A100, but the shape
matters for us: with 250 GiB HBM + 494 GiB host, this is the "experts on host from
load, everything else on GPU" mode in mainline. **Not merged.**

Careful read: this PR's oracle change "select[s] the CPU backend unconditionally" and
"skip[s] the AMX prepack, because the consumer is an out-of-tree CPU engine" — i.e. the
weights land in host and an **external CPU engine computes them**. That is NOT our lane
(we read host experts from the GPU over C2C). It does not replace UVA offload.

## 4. Rival / peer public numbers (new since our campaign — we are no longer the only datapoint)

**SGLang blog: "From 35 to 873 tokens/s" — DeepSeek-V4.1 Flash kernel optimization**
(https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization)
4× GB300, attention TP4, MoE TP4+padding, BS=1, DSpark block 5.
**873.6 tok/s at BS=1** (simulated accept length 5.50), up from 203.3 plain decode.

This is *not* comparable to our 89.6 C1 and must not be presented as such:
- It is a **16-round cumulative kernel-fusion ladder** — FP8 GEMM scale-layout fix alone
  took 35.2 → 117.8 tok/s. Real engineering, but SGLang-specific.
- **4 GPUs vs our 1.** Attention TP4 across the whole model.
- **Acceptance is SIMULATED** (the client always accepts 5 or 6 tokens). They say so:
  *"Acceptance is simulated, so the generated text is not used to judge answer quality;
  simulation mode also disables the in-graph acceptance path."* Real accept length on our
  prose traffic is 1.49 — we would need far more than their loop to see that number.
- Throughput excludes full prefill.
- The ladder's own conclusion is a genuine cross-stack lesson: **"When bringing up a new
  model, checking which kernel a GEMM actually dispatches to is usually worth more than
  tuning tiles."** Their 35→118 step was a quantization *scale-layout* mismatch forcing a
  slow fallback path — exactly the class of bug that would be worth 3× on our box.

**catid / dgx_station_benchmarks — DeepSeek-V4.1-Flash on 2× GB300** (updated Sep 11)
(https://github.com/catid/dgx_station_benchmarks/blob/main/deepseek-v4.1-flash/README.md)
- vLLM PP2 · DSpark: **252.9 tok/s per-user C1**, 3,257.9 agg @C64
- vLLM TP2 · DSpark: 201.0 C1, **3,401.6 agg @C64**
- Prefill: 55,992 prompt tok/s single 128K (vLLM PP2)
- Explicitly: **"One station was not attempted."**

That is the honest two-Station comparison point for our single-Station 89.6 C1 / 312.4 C16.
Note they serve revision `dba1be0a` (the same revision our earlier 2×Station PP2 reference used),
also "more than one GB300 holds".

Also surfaced: several community single-box recipes (sfxnz 2×DGX-Spark EXL3, ZackO2o
4×GB10 1M recipe) — low signal, all multi-box or requantized.

## 5. Kernel integration tracker — DeepSeek's own kernels landing in vLLM

Issue #56217 "[Feature]: DeepSeek-V4.1-Flash Kernels Integration & Optimization Tracker"
(opened 2026-09-10, 0 comments) — tracks DeepGEMM / DeepSelect / FlashMLA into vLLM:

| PR | Item | State |
|---|---|---|
| #56464 | DeepSelect TopK for the DSA sparse indexer | **MERGED 2026-09-13T07:44** |
| #53040 | Fuse shared experts into MegaMoE | merged 2026-08-20 (pre-V4.1) |
| #56266 | Mega-Gate from DeepGEMM | open |
| #56255 | Mega-mHC from DeepGEMM | open |
| #56254 | DeepGEMM sparse MQA logits into V4.1 indexer | open |
| #56344 | V4.1 Attention MegaKernel (FlashMLA #221: fused norm+RoPE+sparse-attn+inv-RoPE+FP8 cast) + `nvfp4_ds_mla` KV | open |
| #56568 | Pad shared experts for native MegaMoE fusion (V4.1's 2304-wide shared expert can't fuse with 2560-padded routed experts) | open |

Upstream kernels: DeepGEMM PR #432 "Public Release 26/09" merged 2026-09-10;
FlashMLA PR #221 (v4.1 kernels) merged 2026-09-10, PR #223 open (CUDA 13.0 build fix).

**Read:** the throughput work for this model is now happening upstream, on main, in a
funded tracker. Every one of these is a decode/prefill win that lands without us.
Our fork sits on the wrong side of that flow.

## 5b. CORRECTION — `dsv41-feat` was replaced, not simply deleted

`dsv41-optimized` **exists**, and its HEAD is **`e47aa780` (2026-09-10T07:23:49Z,
"[Model] Support DeepSeek-V4.1-Flash")** — the *same commit* our pinned image was built from.
One commit on top of Sep-4 main; the rest is the Sep-4 base.

So the true picture is cleaner than "fork of a deleted branch":
- `dsv41-feat` was renamed/replaced by **`dsv41-optimized`**, which is now the **staging branch
  for the remaining kernel work** — #56255 (Mega-mHC), #56266 (Mega-Gate) and #56344
  (Attention MegaKernel) all have `base.ref = dsv41-optimized`.
- The same DSV4.1 support commit went **to main** via #56228 (09-10 12:16Z) and the wiring
  PR **#56214 "[Model] Support DeepSeek-V4.1-Flash" (MERGED 2026-09-11T09:11:20Z, 49 files)**.
- Therefore: **main is now ahead of our image for model support**, while the perf kernel work
  is staged on `dsv41-optimized` and is *not* on main yet.
- Our image tag `deepseekv41-flash-0909` was built 2026-09-10 06:47, i.e. from the same
  `e47aa780` line — so the rebase to main is a move *forward*, not a lateral hop.

## 6. RFC #38256 — still no reply to our three comments

Our comments: 2026-09-12 07:32Z, 11:58Z, 17:06Z (as jmeadlock). Last thread activity after
ours: **none**. Last other comments were MorrisZJ 2026-09-12T00:11Z (@CRPONCELET union-of-experts)
and CRPONCELET 2026-09-06. So: our negative results on coherent-C2C placement are posted
and unaddressed as of 06:40 CDT.

**PR #56177 (01554's expert pool, the #56175 implementation) went STALE:** mergify bot
2026-09-10T15:01 flagged **merge conflicts, rebase required**; last code update
2026-09-13T09:47 is just a rebase churn, no review. Still 29 files / +5,376. Its own
headline remains **7.4 → 63.2 tok/s on a 48 GiB RTX PRO 6000 budget** — a PCIe-class
baseline; we start at 89.6 by reading pinned host memory in place.

## 7. Corrections to our own open-items list

- **NEXT-STEPS item "Hermes-harness 10/10 on v12" is ALREADY DONE**, not open. The recipe
  README cites `results/2026-09-12-v12-1M-k5-off60-util97/harness-summary.md`, v12 re-run
  2026-09-12, **10/10**. My resume summary listed it as outstanding — that was stale.
  Remaining genuinely-open item: the V1 batch-size K-schedule, never booted.

## 8. Literature (in flight)

Six 2026 expert-prefetch/cache papers gathered (2609.04895, 2608.11688, 2607.24787,
2606.15453, 2603.19289, 2511.05814); full assessment running in a subagent. Early read of
the highest-signal one, **2609.04895 (Cache-Aware Joint Router Adaptation, v2 2026-09-10)**:
it formulates expert-cache management as **model-side post-training** — "jointly adapts the
MoE backbone and lightweight auxiliary routers **while preserving the native inference-time
Top-K rule**", reporting +1.15–18.03 pt hit rate and −4.6–53.3% weight traffic on Qwen3 over
the strongest prefetching baseline, task-dependent on GPT-OSS.

Why that matters here specifically: our histogram showed static cold-sets are
**domain-specific** (agent↔prose Spearman 0.155). A method that reshapes routing to *be*
cache-friendly would attack exactly that weakness — but it needs training on our traffic,
which is a different kind of project than anything in the campaign.

## 8. Literature — assessed (subagent, 8 papers)

**Verdict across the board: the actionable work is measurement, not deployment.** Every
runtime mechanism in these papers assumes PCIe/LPDDR H2D copies to hide or a per-expert
GPU cache to fill. We read pinned Grace memory in place at ~340 GB/s with no copies, so
there is nothing to hide — prefetching only helps if it moves bytes into HBM early enough
to need *fewer* total C2C reads, and extra wrong prefetches burn the same bandwidth.

| Paper | Mechanism | Hardware assumed | Training? | Verdict |
|---|---|---|---|---|
| 2609.04895 Cache-Aware Joint Router | post-train backbone + aux routers so routing is cache-retainable, native top-k kept | offloaded MoE w/ explicit cache+transfer | **yes, joint** | test the *diagnostic* only — modifying routing is too invasive |
| 2608.11688 APEX | prefetch router predicts experts pre-attention; adaptive top-(k+δ) | **edge**, PCIe 6.0 x16 + LPDDR | yes | **irrelevant here** — built around transfer hiding |
| 2607.24787 SpecPrefetch | shared adapter predicts next-layer experts for async transfer, frozen router | mobile/edge (Snapdragon) | yes, small adapter | test recall offline only; needs a per-expert cache kernel to deploy |
| 2606.15453 ST-MoE | history + cross-layer correlation tables, reconfigurable HW pipeline | **custom accelerator** | no | irrelevant for implementation; fixed tables brittle across domains |
| 2603.19289 Speculating Experts | quasi-hidden state + default vectors predict next-layer experts | PCIe GPU, copies = 84–88% of TPOT | parameter-free (+ optional estimator) | **best candidate to test offline** — but only as a predictor; overlaps with our DSpark lane |
| 2511.05814 Caching & Pre-fetching study | LRU vs LFU for expert caches + next-layer gate preload | consumer/datacenter PCIe | no | trace analysis only; matches our skew finding, too weak for GB300 |
| 2502.05370 FineMoE | iteration-level expert maps + prompt embeddings → trajectory retrieval | multi-GPU server w/ CPU offload | no | ranking only — can't place experts per-expert on our stack |
| 2505.16056 SRP/SCH | metrics predicting whether expert offload can work | analytical | no | **adopt for evaluation** — formalizes our no-go |

Key sub-findings worth keeping:
- Cost is a **predictor, not the point**: *"the ceiling is bounded — expert fetch is 3–4 ms of
  an 11 ms step, and extra wrong prefetches burn bandwidth."*
- 2505.16056's own numbers support our conclusion independently: *"toy throughput correlations
  are modest (decode overhead correlation about r=−0.3)"*, and our oracle hit rate 0.68 at
  3× top_k already says token-local caches are weak at 384 experts.
- **Speculative execution of experts: do not adopt.** It changes which experts actually run
  (quality risk), and we already attack latency via DSpark k=5.
- The architectural blocker repeats: FlashInfer MXFP4 MoE takes one contiguous
  `[num_experts,…]` tensor, so *no* cache paper can express its cache without the kernel
  change we already priced at 2–3 days + 74-min autotune + parity gate.
- Net: **adopt SRP/SCH + trace-based predictor evaluation** (free, no serving-path change);
  **test quasi-hidden / SpecPrefetch recall offline**; **defer every runtime cache mechanism**
  until a per-expert kernel path exists that can beat direct pinned C2C reads.

## 8b. Rebase feasibility — assessed (subagent)

Mainline DSV4.1 support is real and landed in two steps:
**#56228** (definitions, 09-10 12:16Z) then **#56214 "[Model] Support DeepSeek-V4.1-Flash"
(merged 2026-09-11T09:11:20Z, 49 files, +3,996/−138)** which registered/wired it across
`registry.py`, `config/{engram,kernel,speculative,vllm}.py`, and `gpu_model_runner.py`.

All three dependencies survive with compatible names:
- **UVA offload**: `vllm/model_executor/offloader/uva.py`, `--offload-backend` /
  `--cpu-offload-gb` / `--cpu-offload-params` all still wired in `arg_utils.py`.
- **Engram**: `EngramConfig.cpu_offload` in `vllm/config/engram.py`; #56512 (merged 09-13
  09:44Z) **removes the legacy offload env var** and now defaults V4.1/Qwen4Exp to CPU
  offload — so `--engram-config '{"cpu_offload": true}'` stays valid but becomes redundant,
  and the flag name is stable.
- **DSpark + dynamic SD**: `DSparkModelTypes = Literal["dspark"]` in
  `vllm/config/speculative.py`; `num_speculative_tokens_per_batch_size` +
  `is_dynamic_num_spec_tokens_enabled()` still present.

**Image choice — the catch:** the newest nightly,
`vllm/vllm-openai:nightly-2671fedf…` (09-13 06:15Z, cu129 06:27Z), contains #56228/#56214 but
was cut **before #56512 merged at 09:44Z**. It also predates #56682 (09-13 09:23Z, Engram
CPU-fill startup fix) and #56464 (09-13 07:44Z — actually *before* it, so DeepSelect top-k
IS in that nightly). **v0.29.0 (09-09) predates everything and will not work.**

So: for a first probe, the 09-13 nightly is the right image; if we want the Engram
prefetch/DP-shard work we need the *next* nightly or a source build.

Merged upstream wins already in that image: #56464 DeepSelect top-k for the DSA indexer,
#56562 fuse DSV4.1 input-metadata prep with Triton. Still staged on `dsv41-optimized` and
**not** on main: #56344 (attention MegaKernel — defaults fused attention OFF per its body),
#56568 (shared-expert padding for MegaMoE; +9.6–17.7% on B200 TP8), #56254, #56255, #56266.

Three risks the rebase carries:
1. The nightly predates #56512, so Engram-offload behavior is not the final shape.
2. Tracker #56217 is open — main may boot and serve but **not yet match our pinned fork's
   throughput**, because the fork line carried kernels main doesn't have.
3. #54610 is still **blocked (REVIEW_REQUIRED, mergeable_state=blocked)**, so the default
   offload-order bug persists on main; our explicit `--cpu-offload-params` stays mandatory.
   (Note: #54610's own reproduction is a second independent datum for our finding —
   47.69 GB offloaded, 110.73 ms ITL vs explicit 47.24 GB / 111.64 ms.)

## 9. What this changes

| Finding | Effect on plan |
|---|---|
| `dsv41-feat` deleted; DSV4.1 in main since 09-10 | Our image is a fork of a deleted branch, predating the merge by 6h. **Rebasing is now the highest-value action** — but it is a full re-validate, not a tag bump. |
| Kernel tracker #56217 with funded kernel work on main | The next gains arrive upstream. Staying on the fork means missing them. |
| #54610 independently reproduces our offload-order finding (2.2×) | Our `PARAMS` flag is right; cite it. Adopt the fix on rebase. |
| #56118 lands "experts constructed on host" in mainline | Not our lane (it feeds an external CPU engine), but it is the beginning of mainline accepting host-resident experts. |
| SGLang 873 tok/s is 4-GPU + simulated acceptance | Not a peer number. Do not compare. Borrow the *method*: verify kernel dispatch when bringing up a new model. |
| catid 2×Station 252.9 C1, "one station not attempted" | We remain the single-Station datapoint. Our recipe is still the only published 1×GB300 1M-context config. |
| RFC #38256 silent for 18h on our comments | Nothing to wait for. |
