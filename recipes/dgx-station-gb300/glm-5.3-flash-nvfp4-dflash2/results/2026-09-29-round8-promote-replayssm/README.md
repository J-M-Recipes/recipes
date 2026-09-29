# 2026-09-29 — Round 8: the residual is the instrument floor; ReplaySSM candidate promoted to the daily lane

Follow-up to [Round 7](../2026-09-28-round7-39688-revert/README.md). Same box, weights, flags, gates. Runner [`receipts/cardK3/cardK3.sh`](receipts/cardK3/cardK3.sh); per-arm receipts under [`receipts/cardK3/`](receipts/cardK3/); promotion boot receipts under [`receipts/promote/`](receipts/promote/).

Round 7 left three bars before the reverted-#39688 + ReplaySSM image (C1) could become the daily: (a) a deliberate greedy self-repeat; (b) explain or bound the greedy 1/20 residual vs the W2 reference that persisted with teacher-forced Δ at 0.000; (c) re-measure the Round 5 max-effort regression (−9.5 %, fp32 SSM state). It also left one untested suspect for (b), #39200, because the arm meant to revert it had shipped a duplicate overlay. This round runs all four in one window, with the overlay's #39200 revert **verified in-container by hash before boot** ([`receipts/cardK3/D-mhc-sha.txt`](receipts/cardK3/D-mhc-sha.txt): `communicator_mhc.py` = the pre-`2fa6b94e` blob `f8fd0ac9…`).

## Card K3

| arm | image | spec | greedy vs W2 ref | TF mean / p99 | tools | C1 low hist / code | C1 **max** hist / code | KV pool |
|---|---|---|--:|--:|--:|--:|--:|--:|
| D | `582389ce` + revert #39688 **+ revert #39200** + ReplaySSM | DFlash2 | 1/20 | 0.000 / 0.000 | 10/10 | 197.9 / 300.5 | — | 856,576 |
| F-ar | v0.5.20 release | **AR** (`SPEC=none`) | 1/20 | 0.000 / 0.000 | 10/10 | 138.8 / 137.5 | — | 4,186,752 |
| E | **C1 candidate** (`582389ce` + revert #39688 + ReplaySSM) | DFlash2 | 1/20 | 0.000 / 0.000 | 10/10 | 197.4 / **318.7** | **262.8 / 255.8** | 856,576 |
| E-ctl | v0.5.20 daily | DFlash2 | **20/20** | 0.000 / 0.000 | 10/10 | 200.9 / 286.8 | 263.2 / 260.1 | 577,024 |

Cross-comparisons on the saved greedy JSONs:

| pair | identical | reads as |
|---|--:|---|
| E vs C1 (Round 7) | **20/20** | candidate self-repeat, third boot, deliberate → bar (a) |
| D vs C1 | 1/20 | reverting #39200 *changes* greedy but does not restore the reference → #39200 is not the residual |
| **F-ar vs W2** | **1/20** | v0.5.20's own AR decode vs v0.5.20's own DFlash reference sits at the same floor |
| F-ar vs E | 2/20 | v0.5.20 AR vs the candidate: same band |
| E-ctl vs W2 | 20/20 | the instrument still discriminates when nothing changed |

## What the tables say

**1. The greedy residual is the instrument floor, not a fidelity signal.** `F-ar` is the stock v0.5.20 release image with speculation off; against the v0.5.20 DFlash reference it reads 1/20, the same prompt agrees, and the first-divergence offsets fall in the same 35–250-character band as every reverted arm. The 09-21 rebase bundle already established that DFlash2 is target-verified but not byte-identical to AR on this model (argmax tie-breaking in the verify path). What Rounds 5–7 were reading as "greedy 1/20 on the nightly" is that same tie-breaking, exercised by any change to the kernels behind the verify path — and it is invisible to teacher forcing because teacher forcing does not sample. Bar (b) is met by bounding: with TF at 0.000 to the last digit on five reverted boots, and the reference's own AR at the same 1/20, there is no remaining fidelity claim for the residual to carry. Greedy-vs-a-DFlash-reference is retained as a **same-image regression check** (E-ctl 20/20 proves it works for that) and dropped as a cross-image fidelity gate; the teacher-forced score is the cross-image gate.

**2. #39200 stays.** Reverting it moves greedy without restoring anything and costs nothing to keep; the candidate remains "revert #39688 only". The patch is [`patches/revert-39688-on-582389ce.patch`](patches/revert-39688-on-582389ce.patch) (sha `75eb12b4…`, identical to Round 7's). The both-reverts patch is kept as a receipt.

**3. The max-effort trade is gone.** Round 5 measured −9.5 % thinking-counted C1 on ReplaySSM (fp32 SSM state) against the drifted nightly. On the clean candidate, same window, paired with the v0.5.20 daily: history 262.8 vs 263.2, code 255.8 vs 260.1 — within run-to-run noise. Bar (c) met. (Whether the Round 5 number was the drift or the nightly's other kernels is not separable now and does not matter for the daily.)

**4. The candidate beats the daily on every axis it changes.** KV pool 577K → 857K tokens (+48 %) at the same `mem-fraction-static`; `intermediate_ssm_state_cache` 9.57 GB → 0; code C1 287 → 319 (+11 %); history and max-effort flat; tools 10/10; TF 0.000.

## Promotion (James: y, 2026-09-29 15:50 CDT)

Daily lane `:30001` is now `glmf-DAILY-rssm-20260929`:

- **Image** `glmf-sglang:0922-582389ce-revert39688-tf5.16.1` = `lmsysorg/sglang:nightly-dev-cu13-20260922-582389ce` + transformers 5.16.1 / tokenizers 0.23.2 + [`patches/revert-39688-on-582389ce.patch`](patches/revert-39688-on-582389ce.patch) applied over `/sgl-workspace/sglang/python/sglang` (10 files; `prefill_track_metadata.py` removed). Build recipe in the recipe README.
- **Launch** = the daily config + `--enable-linear-replayssm-spec`. Nothing else changed: same 48 KDA slots, block 7, fp8 KV, mem 0.85, 1M ctx, MAXBS 16.
- **Boot gate, every boot** ([`receipts/promote/`](receipts/promote/)): in-container `sha256sum` of `glm5_next.py` (`bae77af6…`) and `fla/kda.py` (`5d0e0922…`) must match the reverted blobs and `prefill_track_metadata.py` must be absent, or the launcher exits — the patched files are the recipe, and a hook edited after boot silently changes the hash. Then `vision_gate.py`, greedy vs the C1 reference (same-image check), TF vs v0.5.20, tools, C1→C32 warm.
- **Pin discipline:** this is a nightly plus a hand revert, not a tagged release. The next tagged SGLang that carries #40517 gets the same revert and the same gates, and replaces this image if it passes; if upstream fixes #39688 on [sglang#41609](https://github.com/sgl-project/sglang/issues/41609) the revert is dropped and re-gated. Until then, no other upstream movement on this lane.

## What changes in the card / recipe

- Hero C1 code 288 → 319; KV pool 577K → 857K; engine pin `582389ce` (+revert); ReplaySSM added to the guesser/memory line; fidelity strip gains "TF 0.000 vs v0.5.20 on every reverted boot (7)" and reclassifies greedy-vs-DFlash-reference as a same-image check with the v0.5.20 AR floor stated.
- `limits`: Round 5's "max-effort −9.5 %" line struck as `CORRECTED 2026-09-29`; Round 7's "candidate, not promoted" line struck as `PROMOTED 2026-09-29`.

Containers `glmf-K3-{D,F-ar,E,E-ctl}` stopped and kept. `glmf-DAILY-rssm-20260929` left running.

## Reproducibility of the promoted image

The image serving on the Station was built as a `COPY` overlay of the reverted files (Round 7 method). [`Dockerfile.0922-revert39688`](../../Dockerfile.0922-revert39688) rebuilds it from the pinned base digest with `git apply` of [`patches/revert-39688-on-582389ce.patch`](../../patches/revert-39688-on-582389ce.patch). Checked 2026-09-29 on the Station: aggregate sha256 over every `srt/**/*.py` and `kernels/**/*.py` in `/sgl-workspace/sglang/python/sglang` is **`6266d9f1…` in both images**; `prefill_track_metadata.py` absent in both; `pip freeze` transformers 5.16.1 / tokenizers 0.23.2 / torch 2.13.0+cu130 in both. Receipt: [`receipts/dockerfile-reproduce.txt`](receipts/dockerfile-reproduce.txt).
