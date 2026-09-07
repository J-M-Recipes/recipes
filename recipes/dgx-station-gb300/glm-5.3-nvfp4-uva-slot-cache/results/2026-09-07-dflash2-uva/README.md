# DFlash2 over selective-UVA slot cache — stopped at K4

**Verdict: STOP / not promoted.** The candidate booted only with explicit eager execution, but its weighted accepted length was **1.5718**, below the frozen **3.0** stop gate. The C1/C4/C8 campaign, decode-quality campaign, and 512K promotion attempt were intentionally not run.

## Decision table

| Gate | Result | Evidence |
|---|---:|---|
| Target/draft geometry | PASS | `geometry-audit.json` |
| Candidate API bind | PASS on attempt 2 | `candidate-k4-eager-init-extract.log` |
| K4 weighted acceptance | **FAIL: 1.5718 < 3.0** | `acceptance-k4-eager.json` |
| Short-window C1/C4/C8 | NOT RUN | blocked by acceptance gate |
| Teacher-forced / greedy quality | NOT RUN | blocked by acceptance gate |
| 512K DFlash daily-profile promotion | NOT RUN | blocked by acceptance gate |
| Rollback to preserved 512K/MTP | **PASS** | running keeper, authenticated model inventory and exact-marker smoke |

## Fixed candidate

- Target intent: `incoai/GLM-5.3-NVFP4@54e52520606f96b3d9fc84088ad22882a61648ac`
- Draft: `incoai/GLM-5.3-DFlash2@425aa615ce320caac34400208b30808c8f14f76c`
- Draft weight SHA-256: `3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161`
- Image: `vllm-glm53-uva:v0.28.0-2cf0a691`, local image ID `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`
- Candidate: sc13g, 8 GiB bf16 KV, 65,536-token window, sequence cap 4, DFlash K4, explicit `--enforce-eager`
- Acceptance workload: two prose and two code requests, temperature 0, 512 completion tokens each

## Acceptance result

| Kind | Completion tokens | Verification steps | Accepted length | Accepted / drafted | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| prose 1 | 512 | 333 | 1.5375 | 0.1336 | 7.771 |
| prose 2 | 512 | 325 | 1.5754 | 0.1446 | 7.984 |
| code 1 | 512 | 310 | 1.6516 | 0.1637 | 8.359 |
| code 2 | 512 | 335 | 1.5284 | 0.1313 | 7.721 |
| **weighted / median** | **2,048** | **1,303** | **1.5718 weighted** | — | **7.8775 median** |

The gate measures accepted length as completion tokens divided by the delta of vLLM's `spec_decode_num_drafts_total`, matching the predeclared contract. The result is not close to the threshold: the drafter proposed 5,212 tokens and only 745 were accepted across the four requests.

Do not compare the 7.88 tok/s median directly with the historical warmed `bench3.sh` table. The acceptance harness and throughput harness are different regimes. The acceptance result alone terminates the experiment, so no `bench3.sh` number was produced for this candidate.

## Startup history

Attempt 1 loaded the target and draft but failed before API bind during CUDA-graph capture. The slot-cache statistics hook performed an operation unsupported during stream capture, invalidating the graph (`candidate-k4-attempt1-init-extract.log`).

Attempt 2 added only explicit `--enforce-eager`, as recorded in the pre-acceptance amendment to `CONTRACT.md`. It reached `Application startup complete`, resolved `DFlash2DraftModel`, served all four requests, and was then stopped because the acceptance gate failed. `candidate-k4-eager-live-facts.json` records the sanitized command, mounts, image, and non-secret environment.

## Interpretation

This is a negative transfer result for **DFlash2 K4 on this single-GB300 selective-UVA/slot-cache path**. It is not evidence against DFlash2 on HBM-resident GLM-5.3 deployments. The likely mismatch is the one identified before the run: wider verification is expensive when target expert rows are partly served from Grace memory, while the draft's accepted length here is too short to amortize that work.

The result does not establish whether a retrained draft, a different K, or a slot-cache implementation designed around speculative batches could work. Those are new experiments and are not implied by this stopped run.

## Restore proof

The exact preserved pre-experiment container `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907` was restarted rather than rebuilt. At `2026-09-07T18:40:12.250576+00:00`, sanitized inspect recorded it running, not OOM-killed, with 48 GiB bf16 KV, a 524,288-token limit, sequence cap 1, the 5,792-slot map, and native MTP(1). Authenticated `/v1/models` returned `glm-5.3-big` with `max_model_len=524288`; an authenticated temperature-zero completion returned exactly `RESTORE_OK`.

Receipts: `restore-live-facts.json`, `restored-models.json`, `restored-smoke.json`, and `restore-init-extract.log`.
