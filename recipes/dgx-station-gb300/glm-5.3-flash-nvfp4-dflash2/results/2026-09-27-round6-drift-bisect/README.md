# 2026-09-27 — Round 6: bisecting the nightly drift — bracketed to 184 commits (09-18 → 09-21); one hypothesis tested and falsified

Follow-up to [Round 5](../2026-09-27-round5-nightly-replayssm/README.md), same window, same box (`:30001`, GB300 Station), same gates. Question: *which upstream change between v0.5.20 (`94602c9c`, 09-18) and the 09-27 nightly (`425a1f8f`) moves GLM-5.3-Flash's teacher-forced log-probs by 0.139 and breaks greedy parity 20/20 → 1/20?*

Seven boots, gates only (greedy 20 vs the W2 reference, teacher-forced |Δlogp| on the 2,857-token reference vs the v0.5.20 control, tools ×1, plus a `c1_methods` low row so speed is not lost). Runners: [`cardJ_bisect.sh`](cardJ_bisect.sh), [`cardJ2_finalize.sh`](cardJ2_finalize.sh). Receipts per rung in [`receipts/`](receipts/). Each rung is the public `lmsysorg/sglang:nightly-dev-cu13-*` image with transformers 5.16.1 derived on top, exactly as the release image.

| rung | image (main commit) | commits after v0.5.20 | greedy | TF mean \|Δlogp\| | TF p99 | tools | C1 hist / code |
|---|---|--:|--:|--:|--:|--:|--:|
| ctl | v0.5.20 `94602c9c` | 0 | 20/20 | 0.000 | 0.000 | 10/10 | 201.5 / 287.4 |
| **r5** | 09-18 `20518d85` | 66 | **20/20** | **0.000** | 0.000 | 10/10 | 201.5 / 287.4 |
| r1 | 09-21 `0f6761b5` | 314 | 1/20 | 0.139 | 1.93 | 10/10 | 201.9 / 302.8 |
| r2 | 09-22 `582389ce` | | 1/20 | 0.139 | 1.93 | 10/10 | 202.8 / 304.5 |
| r3 | 09-23 `06008c17` | | 1/20 | 0.139 | 1.93 | 10/10 | 201.3 / 301.6 |
| r4 | 09-25 `8ca82118` | | 1/20 | 0.139 | 1.93 | 10/10 | 199.9 / 299.2 |
| r6 | 09-27 `425a1f8f` + `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1` | | 1/20 | 0.139 | 1.93 | 10/10 | 203.4 / 305.4 |
| r7 | r6 + `--enable-linear-replayssm-spec` | | 1/20 | 0.139 | 1.93 | 10/10 | **222.9 / 322.7** |

Three things the table says plainly.

**1. The drift enters in one step and then never changes.** Every nightly from 09-21 through 09-27 gives *bit-identical* TF statistics (mean 0.1391908979…, p99 1.9313104748…). That is one deterministic change to the forward pass, not accumulated kernel churn. ~~It means the +6–10 % code/prose speedups that arrive later in the week (the mHC-boundary fuse, #39200, lands at commit 218 of 314 in the 09-21 window and the KDA-projection fuse #39688 at 183) are *not* what moved the numerics; they are riding on top of an earlier change.~~ **CORRECTED 2026-09-28:** that inference was wrong. "Commit N of 314" is an ordinal inside the window, not a position relative to the drift — the whole 184-commit window is one rung, so *every* commit in it, including #39200 (`2fa6b94e`, verified `git merge-base --is-ancestor 2fa6b94e 0f6761b5`, and *not* an ancestor of the clean `20518d85`) and #39688 (`c8eb54c4`), is a candidate. A bisect-by-reading of the window (Round 7) ranks #39688 first: it flips `Glm5NextLinearAttention.do_fuse_qkvbfg` from `quant_config is None` to allow `modelopt_fp4` when the attention projs are unquantized — exactly this checkpoint, where NVIDIA quantized only the routed experts — and moves the raw-beta sigmoid into the Triton gate kernel. #39200 declines above 16 tokens, so it cannot by itself explain the 2,857-token teacher-forced number.

**2. The window is 09-18 → 09-21: 184 commits.** `20518d85` (commit 66) is clean; `0f6761b5` (commit 314) is not. Docker Hub has no `nightly-dev-cu13` image for 09-19 or 09-20, so going further needs source builds. Of the 184 commits, ~43 touch kernels, quantization, MoE, spec or the GLM/KDA path; the ones I would test first, in order:

- `c46bf5e9` **#40105** [MoE] Disable FlashInfer fused finalize by default for numerical accuracy — *tested below; not it*
- `7714b182` #40187 Fix top-1 MoE routing with non-unit scaling
- `42875bcd` #38932 fix(modelopt): dispatch NVFP4 MoE on the cached backend, not the live global
- `f447bb70` #39680 Coalesce the KDA CuTe DSL decode state transpose (claims bit-identical)
- `248c202b` #39859 Use runtime token widths for Triton speculative verification
- `c3aa09b0` #40208 Fuse hc_combine_norm for mid-size verify batches
- `c8eb54c4` #39688 Fuse GLM-5.3-Flash KDA projections and prefill metadata
- `2fa6b94e` #39200 Fuse the glm5_next mHC attn→MLP boundary
- `f65c70bb` #40033 Move CUDA/ROCm speculative kernels to JIT
- `1b200ffa` #40039 Serve 32-wide-K ue8m0 block-FP8 linears through FlashInfer MXFP8 GEMMs

**3. Hypothesis #40105 is falsified.** #40105 flipped `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` from `True` to `False` "for numerical accuracy" at commit 87 — squarely inside the window, MoE-numerics-shaped, and a one-line default change. r6 set the env back to `1` on the 09-27 image: TF unchanged to the last digit. Either the flag is not on the NVFP4 CUTLASS path this model takes, or the drift is elsewhere. Recorded so nobody re-tests it.

## ReplaySSM, re-confirmed (r7)

Same as Round 5, on the same image: boots with DFlash, `KV Cache … #tokens: 856576` vs 577,024 (+48 %), C1 history 222.9 (+10.6 % vs control) and code 322.7 (+12.3 %), TF *identical* to the same image without it. ReplaySSM is not the drift. It remains the single best thing waiting on a clean image.

## What this changes

- **Pin stays v0.5.20.** Nothing here is adoptable; the drift is in the image, and the image is the only way to get ReplaySSM today.
- The **09-18 nightly `20518d85` is a proven-clean alternate base** (20/20, 0.000) if a derived image is ever needed — but it has neither #40517 nor the fusions, so it buys nothing over v0.5.20.
- Next step, if the +48 % pool / +11 % C1 matters enough: **source-build bisect** on the 184-commit window (a `pip install -e` of `python/` inside the 09-18 image is enough for Python-side changes; kernel changes need the sgl-kernel wheel, which is the expensive half). Or — cheaper — file the TF numbers upstream and ask which of the ~10 candidates above changed GLM-5.3-Flash numerics on SM100 NVFP4; the maintainers can answer that from the diff faster than we can from boots.

Containers `glmf-J-r1…r7` stopped and kept. `:30006` DSV4.1 v21 restored after the window.
