# Upstream watch — 2026-09-20 (evening)

Read-only sweep of the pinned SGLang stack against upstream on 2026-09-20. No box changes.

## Pinned vs upstream

| Component | We pin | Upstream now | Delta |
|---|---|---|---|
| SGLang image | `lmsysorg/sglang:nightly-dev-cu13-20260911-00143e9c` (main 2026-09-11 00:22Z) | **`v0.5.20`** released 2026-09-18 (`94602c9c`; `v0.5.20-cu130` digest `sha256:b27fce60bc54…`); nightly `nightly-cu134-20260920-efa7be2` | pin → v0.5.20 = **220 commits**. v0.5.20 is the first *tagged* release with GLM-5.3-Flash listed as a supported model (#36507, #38621) |
| Target model | `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e` | HF main `09b04e5e` (2026-09-10, "Add SGLang serving instructions") | none |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2@7d74cdd8` ("Release", 2026-08-27) | HF main `bf582e4e` — **two "Checkpoint update" commits after our pin** (`dc77ff1c` 2026-08-28, `bf582e4e` 2026-08-31); 1.17B BF16 params, 7.0 GB | Our pin is the draft we measured (acceptance 0.17 prose / 0.62 math). A retrained draft is a **new acceptance axis**, not a drop-in: same target, different drafter, needs its own acceptance + TF gate. The box path `/models/GLM-5.3-Flash-DFlash2` is under `/home/exx` (not traversable by `milo`); confirm which revision is staged before the window. |

## What landed between the pin and v0.5.20 that touches this recipe

| # | PR | Merged | In our pin? | Impact | Action |
|---|---|---|---|---|---|
| 1 | **#37818 "Track DFlash Mamba state at checkpoint boundaries"** (fixes #37817): DFlash could miss a linear-attention (KDA) state checkpoint when accepted verify tokens cross a Mamba tracking boundary — full-attention KV and linear-attention state end up at different positions. Implemented in `DFlashWorkerV2`. | 2026-09-12 | **No** (pin is 58 behind) | This is a **correctness** bug in exactly our configuration: GLM-5.3-Flash (KDA layers) + DFlash2 + long generations. It would present as occasional divergence after many accepted tokens, invisible to short greedy-parity prompts. Our teacher-forced gate was run on the pinned image and would not see a state-position skew that only appears on long outputs. | **Window-only rebase to `v0.5.20-cu130`, then re-run the TF divergence gate with long (≥2K-token) generations.** Until then, label the pinned image "known DFlash/KDA checkpoint bug upstream-fixed in v0.5.20" in `limits`. |
| 2 | #37069 TP>1 Domino rollout for DFlash V2 | 2026-09-12 | No | TP1 here — not our path. | none |
| 3 | #39366 "stop shadowing the DSpark shared-experts fusion guard" | 2026-09-15 | No | DSpark, not DFlash — not our path. | none |
| 4 | #36899 optimized Domino rollout DFlash V2; #38522 opt-in breakable prefill CUDA graphs for GLM-5.3-Flash (`--cuda-graph-backend-prefill breakable`) | 09-09 / 09-10 | **Yes** | Already in our measurements. Breakable prefill graph was not tested here; cookbook now exposes it. | PARKED — candidate for a prefill-latency window, not decode. |
| 5 | v0.5.20 notes: Unified radix tree branching-point caching for SWA (DSV4-Flash TTFT 1.57→1.07 s on shared prefix); SGLang Simulator (CPU-only scheduler/cache model). | 09-18 | No | Radix change may move our TTFT p95 under shared system prompts; Simulator could size `--max-mamba-cache-size` offline instead of by boot. | Measure on the rebase; try Simulator for the KDA-slot budget. |
| 6 | Cookbook (docs.sglang.io GLM-5.3-Flash) now recommends computing `--mamba-full-memory-ratio` / `--max-mamba-cache-size` via the repo-local `compute-mamba-ratio` skill from one boot log's pool sizes — the exact knob our Round 2 "7-concurrent cap" finding was about. | — | — | Upstream now documents our finding's mechanism. | Cite in README; our pinned values stand. |

## Bottom line

One correctness item (#37818) argues for a **scheduled rebase window** to `v0.5.20-cu130` with the long-generation TF gate, ahead of any new speed work on this recipe. Everything else is measure-on-rebase. The draft checkpoint moved twice upstream after our pin; the newer draft is a separate, gated axis for the same window, never a silent swap.

## Sources

sgl-project/sglang releases `v0.5.20`; PRs #37818, #37069, #39366, #36899, #38522, #38621; issue #37817; docs.sglang.io GLM-5.3-Flash cookbook; Docker Hub `lmsysorg/sglang` tags; HF `nvidia/GLM-5.3-Flash-NVFP4`, `incoai/GLM-5.3-Flash-DFlash2` API.
