# 2026-09-28/29 — Round 7: the drift is #39688; ReplaySSM is clean on a reverted 09-22 image (candidate, not promoted)

Follow-up to [Round 6](../2026-09-27-round6-drift-bisect/README.md). Same box (`:30001`, GB300 Station), same weights (`nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e`, draft `7d74cdd8`), same daily flags, same gates (greedy 20 vs the W2 reference, teacher-forced |Δlogp| on the 2,857-token reference vs the v0.5.20 control, tools ×1, `c1_methods` low). Three cards, unattended overnight, nine boots. Runners: [`receipts/cardK/cardK_revert39688.sh`](receipts/cardK/cardK_revert39688.sh), [`receipts/cardK2/cardK2_prize.sh`](receipts/cardK2/cardK2_prize.sh), [`receipts/cardL/cardL_flood.sh`](receipts/cardL/cardL_flood.sh). Per-arm receipts under [`receipts/`](receipts/); the exact source overlays that were `COPY`'d over `/sgl-workspace/sglang/python/sglang` in each image under [`overlays/`](overlays/).

**Question.** Round 6 bracketed the teacher-forced drift to the 184 commits between the 09-18 nightly (`20518d85`, clean) and the 09-21 nightly (`0f6761b5`, drifted) and falsified #40105. A bisect-by-reading of that window ranked sgl-project/sglang#39688 (`c8eb54c4`, "Fuse GLM-5.3-Flash KDA projections and prefill metadata") first, because it flips `Glm5NextLinearAttention.do_fuse_qkvbfg` from `quant_config is None` to allow `modelopt_fp4` when the attention projections are unquantized — exactly this checkpoint. Does reverting it restore v0.5.20 numerics, and if so, does ReplaySSM (#40517) then pass the gate?

## Method: overlay, not rebuild

`git revert --no-commit c8eb54c4` applies cleanly (runtime files) on `0f6761b5` and on `582389ce`; on `425a1f8f` it conflicts in `glm5_next.py` and `hybrid_linear_attn_backend.py`, and a naive overlay fails import (`retraction_backup` was renamed upstream after 09-22). Each arm is the public `lmsysorg/sglang:nightly-dev-cu13-*` image + transformers 5.16.1 (as the release image) + a `COPY` of the reverted files over the editable checkout + `rm` of `prefill_track_metadata.py` (added by #39688), with an import check at build time. sglang's version string in-container is unchanged, so `receipts/*/version.txt` records the sha256 of the overlaid files instead.

Arm B ("gate-only") keeps the whole commit and only restores the old fusion gate (`_can_fuse_proj` returns False for any quantized model), to separate the two mechanisms inside #39688: unfused projections vs the new Triton `tl.sigmoid` raw-beta path in `kda_gate_chunk_cumsum` + the prefill-track metadata plumbing.

## Card K — 09-21 image

| arm | image | greedy vs ref | TF mean \|Δlogp\| | TF p99 | tools | C1 hist / code |
|---|---|--:|--:|--:|--:|--:|
| ctl | `0f6761b5` unmodified | 1/20 | 0.1391908979 | 1.9313 | 10/10 | 200.6 / 301.7 |
| **A** | `0f6761b5` + **full revert of #39688** | 1/20 | **0.0** | **0.0** | 10/10 | 197.8 / 288.7 |
| B | `0f6761b5` + gate-only | 2/20 | 0.1378 | 2.0970 | 10/10 | 200.0 / 314.8 |

## Card K2 — 09-22 image (`582389ce`, the first nightly that carries #40517 ReplaySSM)

| arm | image | flags | greedy vs ref | TF mean | TF p99 | tools | KV pool (tokens) | SSM cache | C1 hist / code |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| **C1** | `582389ce` + revert #39688 | `--enable-linear-replayssm-spec` | 1/20 | **0.0** | **0.0** | 10/10 | **856,576** | 0.00 GB | 197.1 / **318.5** |
| C2 | `582389ce` + revert #39688 (**intended** + revert #39200 — **did not apply**, see below) | `--enable-linear-replayssm-spec` | 1/20 | 0.0 | 0.0 | 10/10 | 856,576 | 0.00 GB | 196.6 / 317.0 |
| C1n | `582389ce` + revert #39688 | (none) | 1/20 | 0.0 | 0.0 | 10/10 | 577,024 | 9.57 GB | 197.6 / 288.1 |

Cross-comparisons run after the fact on the saved greedy JSONs (`greedy_equiv.py --compare`):

| pair | identical |
|---|--:|
| A (0921+revert) vs C1n (0922+revert) — two images, two boots | **20/20** |
| C1 vs C2 — a same-image reboot (C2's overlay was byte-identical to C1's) | **20/20** |
| A vs C1 — with/without ReplaySSM | 1/20 |
| v0.5.20 ctl2 (Round 5) vs W2 reference | 20/20 |
| v0.5.20 ctl2 vs A | 1/20 |
| 09-27 night+RSSM (Round 5) vs C1 | 1/20 |

## What the tables say

**1. The teacher-forced drift is #39688, and it is the kernel half, not the fusion gate.** Full revert → 0.000 to the last digit, on two different nightlies. Gate-only (projections unfused, everything else kept) → 0.1378, i.e. essentially the same drift. So the number that moved is inside `kda_gate_chunk_cumsum` (`torch.sigmoid` → `tl.sigmoid` on `beta_is_raw`) and/or the prefill-track metadata path, not the merged GEMM. Round 7's bisect-by-reading had the right commit for the wrong reason.

**2. ReplaySSM passes the teacher-forced gate on a clean image.** C1 = 0.000 / p99 0.000, KV pool 577K → 857K (+48 %), `intermediate_ssm_state_cache` 9.57 GB → 0, C1 code 288 → 318 (+10.6 %), history flat (197 vs 198). Same shape as Round 5, now with the target forward matching v0.5.20 token-for-token under teacher forcing. Tools 10/10.

**3. The greedy residual is a second, decode-only change; #39200 is NOT yet ruled out.** With TF at 0.000, greedy vs the W2 reference stays 1/20 on every reverted arm. A and C1n are 20/20 identical *to each other* across two images and two boots, so the reverted forward is deterministic; it just is not the v0.5.20 forward under DFlash sampling. **C2 was meant to also revert #39200 and did not:** the in-container `communicator_mhc.py` sha (`ff29a9ca…`, [`receipts/cardK2/C2/version.txt`](receipts/cardK2/C2/version.txt)) matches stock `582389ce`, and the overlay shipped for C2 was byte-identical to C1's — the second `git revert` was silently dropped when the overlays were staged. So C1 vs C2 20/20 is a *self-repeat of C1 across two boots*, which is useful (promotion bar (a) is half met), but it says nothing about #39200. Reverting #39200 alone on `582389ce` applies cleanly ([`patches/`](patches/) will carry it when it is actually run). Remaining in-window candidates for the residual: #39200 (`2fa6b94e`, mHC fuse, ≤16 tokens — decode-shaped, exactly where the residual lives), #39695 (indexer/Q-quant stream overlap), #39680 (KDA CuTe DSL decode-state transpose), #39859 (Triton spec-verify token widths). The 09-21 rebase bundle already showed DFlash is target-verified, not byte-identical to AR, on this model — so a 1/20 greedy with 0.000 TF may be the instrument's floor for a DFlash-vs-DFlash comparison across images rather than a fidelity failure. That is the open question, and the reason this is a candidate and not the daily.

**4. #39688's speedups are real but modest here.** Unreverted vs reverted on the same image: code 301.7 → 288.7 (−4 %), history flat. The +11 % on code in C1 is ReplaySSM, not #39688.

## Disposition

- **Pin stays `v0.5.20` (`94602c9c`) for the daily lane.** Nothing here is promoted.
- **Candidate image:** `glmf-sglang:0922-582389ce-revert39688-tf5.16.1` (built from `lmsysorg/sglang:nightly-dev-cu13-20260922-582389ce` + transformers 5.16.1 + [`overlays/C1-revert-39688/`](overlays/C1-revert-39688/)), launched with the daily flags + `--enable-linear-replayssm-spec`. Stopped-and-kept on the Station as `glmf-K2-C1`.
- **Promotion bar (next window):** (a) C1 greedy self-repeat 20/20 across two boots — **met by accident** (C1 vs C2 20/20, same image, two boots); re-run once deliberately; (b) explain or bound the 1/20 residual — either find the second in-window commit, or show v0.5.20 DFlash vs v0.5.20 AR greedy already sits at the same floor so the instrument, not the image, is the limit; (c) the Round 5 max-effort regression (−9.5 %, fp32 SSM state) re-measured on C1 before it becomes the daily for thinking-heavy traffic.
- **Upstream:** filed as [sgl-project/sglang#41609](https://github.com/sgl-project/sglang/issues/41609) with the Round 6 table; Card K table added as a comment; the C1 result and the C2 correction as a further comment.

## Card L — MiMo-style flood fixture on the daily lane (agent-card row)

Trigger: Xiaomi's 2026-09-27 postmortem of MiMo-V2.6 tool-call repetition (reward blind spot; flooding penalty only fired above 32 calls/turn; fixed by a 12-step repetition-specialized teacher merged via MOPD). Our MiMo recipe measured RL weights **25/36** history-primed tasks flooded vs MOPD **0/36**. Same fixture, same script bytes (`flood_fixture.py` sha `0c33d56c…`, [`receipts/cardL/fixture-sha.txt`](receipts/cardL/fixture-sha.txt)), run on the **v0.5.20 daily image**, DFlash2 on and off because sgl-project/sglang#40843 (DFlash repetition loops on GLM-5.3-Flash) is open against this exact draft pair.

| arm | tasks | solved | flooded | within-turn rep | cross-turn rep | tasks w/ any dup | max calls / turn | mean calls / task | wall |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DFlash2 on (daily) | 72 | **72** | **0** | 0.0 | 0.0 | 0 | 13 | 4.72 | 679 s |
| AR (`SPEC=none`) | 72 | **72** | **0** | 0.0 | 0.0 | 0 | 13 | 4.71 | — |

Both history-primed halves (h0/h1, 36 each) 36/36, 0 flooded. Metric definitions as in the MiMo bundle: within-turn = Xiaomi's `(N−U)/N` after JSON-canonicalised tool+args; cross-turn = identical call re-issued after an answered call with unchanged environment state; flooded = MAX_CALLS 120 exceeded. No MiMo-class behaviour on this checkpoint in this fixture. This does not speak to the thinking-degeneration reports (#36669) or the DFlash `资料` loop (#40843), which are different failure shapes; a pass here is a pass on *this* axis only.

## Corrections this round forces

- Round 6, point 1: struck in place on 2026-09-28 — "#39200 lands after the drift" was wrong; the whole window is one rung. #39200 remains a live candidate for the greedy residual (C2 did not test it; see point 3).
- The C2 arm as *run* is a C1 self-repeat, not a #39200 test. Recorded as such; the "#39200 ruled out" wording that briefly existed in the recipe README and upstream comment is withdrawn.
- The bisect-by-reading's stated mechanism for #39688 ("fusion gate flip") is wrong; the gate-only arm falsifies it. Recorded here so nobody re-tests the gate.

Containers `glmf-K-{ctl,A,B}`, `glmf-K2-{C1,C2,C1n}`, `glmf-L-{dflash,ar}` stopped and kept. Nothing restored.
