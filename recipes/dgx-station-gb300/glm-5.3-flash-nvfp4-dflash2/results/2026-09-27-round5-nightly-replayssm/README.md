# 2026-09-27 — Round 5: a decode-step profile, the September-27 nightly, and ReplaySSM (W5 reopened) — measured, not adopted

One window on the GB300 Station, `:30001`. Release image `glmf-sglang:0.5.20-tf5.16.1` (v0.5.20 + transformers 5.16.1), NVIDIA NVFP4 @ `09b04e5e`, draft `7d74cdd8`, DFlash2 block 7, 48 KDA slots, mem 0.85, 1M ctx. Five fresh boots bracketing the candidates: **CTL1 → NIGHT → NIGHT+RSSM → CTL2**, plus a profiler boot first. Runner [`cardI_glmf_runner.sh`](cardI_glmf_runner.sh); the correctness gates re-ran on the stopped containers in [`cardIb_gates.sh`](cardIb_gates.sh) after the first pass lost them to an unset `API_KEY` env (speed rows were unaffected). Receipts in [`receipts/`](receipts/).

**Verdict: pin stays on v0.5.20.** The nightly is faster on every class and ReplaySSM is a real, clean capacity win — but the nightly's kernels change the model function (greedy 1/20, teacher-forced mean |Δlogp| 0.139 vs 0.0 on both controls), which is the same failure class that reverted DSV4.1's v19. Not adoptable until a tagged release carries #40517 without the drift, or the drift is bisected to a PR we can exclude.

## What is new upstream since the v0.5.20 pin

| merged | PR | what | our path |
|---|---|---|---|
| 09-20 | #39200 | fuse the `glm5_next` mHC post→pre boundary (90 per forward, 3 launches → 1) | yes — 90 launches / 0.93 ms per step in our profile |
| 09-21 | **#40517** | enable ReplaySSM for GLM-5.3-Flash (gate was `KimiLinearConfig`-typed; now capability-based) | **reopens W5**, rejected at boot on 09-15 |
| 09-24 | #39816 | Cute-DSL all-reduce fusion for DeepseekV2 archs | TP1 — no all-reduce; not our path |
| 09-25 | #41194 | plan NextN/MTP draft layers as one-layer models | native MTP — closed at W7 |
| open | #41105 | capture the KDA extend in breakable prefill CUDA graphs (eager prefill is CPU launch-bound, ~2,300 launches) | prefill / TTFT lever, watch |
| open | #36683 | ReplaySSM spec-verify for DFlash / DSpark on hybrid GDN | note: NIGHT+RSSM booted *with* DFlash on `425a1f8f` anyway — the request-pool wiring from #40517 was enough |
| open | #36830 | fp8 MLA KV blocked by `index_kpool=4` | still the largest capacity item; unchanged |

Drafter, NVFP4 and base checkpoints: no new revisions.

## I0 — first profile of a GLM Flash decode step (v0.5.20, C1, `effort=low`)

[`profile/v0520-prose-C1.pt.trace.json.gz`](profile/) via `/start_profile` (`num_steps=60`), analysed with [`prof_step.py`](prof_step.py). Median step **13.9 ms wall, GPU busy 111 % of wall** — the DFlash2 draft runs on a second stream and overlaps the target (tid 7770: 181 kernels/step). Inter-step gap 1.06 ms.

| | ms / step | launches | note |
|---|--:|--:|---|
| expert GEMMs (`bmm_E2m1…` gemm1 + gemm2) | 4.1 | 84 | 61 + 34 µs each on `t128x8x512` at M ≈ 8 — tile-quantized, same shape as DSV4.1 |
| dense GEMM (nvjet splitK, cutlass TGV, deep_gemm prenorm) | 6.1 | ~300 | attention/KDA projections, drafter |
| **kernels under 12 µs** | **6.6** | **1,505** | of 1,769 launches; `splitKreduce` ×192, mHC pre/post ×180, prenorm GEMM ×90, TGV ×94, routing/finalize ×84 |
| KDA (`fused_sigmoid_gating_delta_rule_update`, `layer_norm_gated`, conv1d) | 1.9 | 102 | linear attention state update |
| mHC (`mhc_pre_big_fuse`, `mhc_post`) | 0.93 | 180 | what #39200 fuses |
| DSA attention (`fmhaSm100f…`) | 0.08 | 11 | negligible at this prompt length |

Reading: like DSV4.1 after residency, the GLM Flash step is **launch-shaped**, not bandwidth-shaped — 85 % of launches are sub-12 µs kernels worth 6.6 ms of a 13.9 ms step. That is why #39200 (and any further fusion) is worth chasing, and why the code/prose classes gained 6–10 % on the nightly below.

## The window

Speed rows: `c1_methods.py` median of three, `reasoning_effort=low`, answer-only; C8 from `flash_bench.knee` (3 reps after warm; CTL2's first C8 leaked a warm pass at 116 % spread and was re-run per the recipe rule).

| | recipe C1 (history 512) | prose 300 | code 400 | shell 200 | C1 max-effort (thinking counted) | C8 agg | KV pool | tools |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CTL1 v0.5.20 | 201.5 | 150.8 | 287.7 | 160.5 | 264.3 | 694 | 577,024 | 10/10 ×2 |
| NIGHT `425a1f8f` | 203.0 (+0.7 %) | **166.3 (+10 %)** | **304.9 (+6 %)** | 167.4 (+4 %) | 265.6 | 736 (+4 %) | 577,024 | 10/10 ×2 |
| **NIGHT + ReplaySSM** | **223.9 (+11 %)** | 159.3 (+6 %) | **324.2 (+13 %)** | 165.1 (+3 %) | **239.3 (−9.5 %)** | **757 (+7 %)** | **856,576 (+48 %)** | 10/10 ×2 |
| CTL2 v0.5.20 | 201.5 | 150.7 | 287.4 | 160.7 | 263.9 | 715 | 577,024 | 10/10 ×2 |

Controls agree to 0.1 tok/s on every C1 class. ReplaySSM frees the 9.57 GB `intermediate_ssm_state_cache` (boot log: `intermediate_ssm_state_cache size: 0.00GB`, ring buffers 2.5 GB) and the KV pool grows 577K → 857K tokens on the same `mem-fraction-static`. It also forces `--mamba-ssm-dtype float32` ("cached replay uses compensated checkpoint projection"), which is the likely cause of the max-effort regression: long thinking generations pay the fp32 state path every step while short answer-only ones gain from the smaller working set. That trade needs its own measurement before ReplaySSM is a daily-lane candidate even on a clean image.

## Fidelity — why nothing is adopted

Greedy 20-prompt capture vs the W2 reference and teacher-forced |Δlogp| on the 2,857-token reference text vs the v0.5.20 control (`tf_noninferiority.py`):

| | greedy identical | TF mean \|Δlogp\| | TF p99 |
|---|--:|--:|--:|
| CTL2 v0.5.20 | **20/20** | **0.000** | 0.000 |
| NIGHT | 1/20 | 0.139 | 1.93 |
| NIGHT + ReplaySSM | 1/20 | 0.139 | 1.93 |
| NIGHT vs NIGHT+ReplaySSM (greedy, each other) | 2/20 | — | — |

The two nightly arms carry **identical** TF numbers, so ReplaySSM itself adds no drift — the change is in the image's kernels, exactly as with vLLM's DSV4.1 nightlies after #56633. Whether it is #39200's fused mHC boundary, the Cute-DSL refactor, or something else between `94602c9c` and `425a1f8f` is a bisect we did not run today. Tools passed 10/10 on every arm, which is the same lesson as before: the tool harness does not see this class of change; the teacher-forced gate does.

## What would make this adoptable

1. **Bisect the drift** between v0.5.20 and `425a1f8f` with the TF gate (four or five nightlies, ~10 min each on stopped-container restarts). If it is a single PR, a derived image with that PR reverted is a candidate.
2. **ReplaySSM on a clean image** — measure the max-effort trade (fp32 SSM dtype) on real agent transcripts, not just the four C1 prompts; if the batch lane (`SPEC=none`, 640 slots) is the target, the +48 % pool alone may justify it there.
3. **fp8 MLA KV (#36830)** remains upstream's move.

Containers `glmf-I-{prof,ctl1,night,night-rssm,ctl2}` stopped and kept; `:30006` DSV4.1 v21 restored after the window.
