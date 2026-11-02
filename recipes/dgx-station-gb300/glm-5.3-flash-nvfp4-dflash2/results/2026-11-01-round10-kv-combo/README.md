# Round 10 (November 1–2, 2026): 2.5× the KV pool and 12 running requests, three flags, no code

One DGX Station GB300 (GPU pinned by UUID; the in-box RTX PRO 6000 is not used). Daily image `glmf-sglang:0922-582389ce-revert39688-tf5.16.1`, nvidia NVFP4 `09b04e5e`, DFlash2 `7d74cdd8` block 7, ReplaySSM, fp8 target KV, 1M context, 48 KDA slots.

## Why
The 857K-token pool looked starved. Boot logs say why: the target's KV is already fp8 (~7,083 B/token), but the DFlash2 draft's KV was bf16 (20,480 B/token = **74% of all KV bytes**). `--speculative-draft-attention-backend fa4` forces the draft KV to bf16; any fp8-capable backend inherits the target's fp8. Separately, the 9-request cap is KDA state slots (3 base + 2 ping-pong per request), not KV.

## Card KV1 — one axis per arm, same window ([`cardKV1.sh`](cardKV1.sh))
| arm | flags vs daily | KV pool (tokens) | running cap | TF / tools / vision | C1 history / code | C8 agg | accept len |
|---|---|---|---|---|---|---|---|
| ctl | — | 856,576 | 9 | 0.000 / 10/10 / pass | 197.4 / 318.7 | 702 | 3.36 |
| T | `--speculative-draft-attention-backend triton` | 1,362,880 | 9 | same | 199.3 / 315.3 | 736 | 3.51 |
| L | `--mamba-radix-cache-strategy extra_buffer_lazy` | 856,576 | **12** | same | 197.6 / 318.8 | 709 | 3.35 |
| M | `--mem-fraction-static 0.90` | 1,341,568 | 9 | same | 195.8 / 314.6 | 705 | 3.36 |
| **COMBO** | T + L + M | **2,134,528** | **12** | same | 200.5 / 317.5 | 702 | 3.58 |
| ctl2 | — | 856,576 | 9 | same | 197.6 / 319.0 | 707 | 3.37 |

C8 spread across reps 2–12%: decode speed is unchanged. Greedy vs the September DFlash reference is 7/20 on the fp8-draft arms (draft KV numerics shift which draft tokens win ties; teacher-forced Δ stays 0.000, so the target's outputs are unchanged). 960K-token needle: found on T, M and COMBO; impossible on the 857K pool. 6×200K concurrent: 3 run at once on 857K (93% KV), 5 on the larger pools.

## Prefix reuse (cards PC, PC2, PC3)
Agent-shaped second turn (`[user, assistant, user]`) reuses **99.8–99.97%** of the prefix at 32K, 64K, 128K and 200K, interleaved (3×32K, 6×32K, 3×150K), streamed or not, with or without ReplaySSM, with or without DFlash. 200K second turn: **19.9 s → 0.61 s**. KDA checkpoint slots peaked at 8%. An earlier 150K retention probe in KV1 read 0% reuse; three follow-up cards could not reproduce it and it is treated as a probe artifact — not cited.

## RTX PRO 6000 cold-expert band: rejected (card ED3)
SGLang's expert-distribution recorder over the synthetic fixtures (ranking) and a 24-turn Hermes transcript replay (holdout). Routing is flat: the hottest 64 of 288 experts carry 55% of decode picks, 128 carry 78%. Moving 64 cold experts per layer to the 6000 would hand it 10.5% of the expert work but put a PCIe round trip on **93% of (step, layer) pairs and 100% of decode steps** (64% even with a perfect hindsight ranking): each DFlash verify step makes 56 expert picks per layer per request. The DeepSeek-V4.1 sidecar pays because its cold experts compete with Grace memory; here they compete with GB300 HBM, and COMBO already supplies the capacity. Measured with `--moe-runner-backend flashinfer_cutlass` — see below.

Two recorder bugs on this image: (1) under DFlash, `_InfiniteBuffer.append` writes into an inference tensor outside InferenceMode and the scheduler dies (card ED) — local fix [`patches/expert-distribution-inference-mode.patch`](../../patches/expert-distribution-inference-mode.patch), recorder boots only; upstream added a growth guard for this mechanism in sgl-project/sglang#40666 (2026-10-12, after this pin; not tested here); (2) with the `flashinfer_trtllm` runner the BYPASSED top-k routes inside the kernel and never reaches the recorder hook, so dumps are all zeros with no warning (card ED2: 5,208 steps, sum 0) — ED3 re-ran with `flashinfer_cutlass`. Clean repro on the unmodified image with speculation off (card RP, 2026-11-02): `flashinfer_trtllm` (4009, 45, 288) sum 0 vs `flashinfer_cutlass` 42 layers populated. Filed as [sglang#47932](https://github.com/sgl-project/sglang/issues/47932).

## Promoted (November 2, 2026, 05:20 CST)
`glmf-DAILY-combo-20261101` on :30001. Patch gate pass, vision pass, greedy 20/20 vs KV1 COMBO, TF 0.000 / p99 0.000, tools 10/10, C1 199.4 history / 315.9 code, pool 2,134,528, cap 12. Rollback: the Sept-29 flags (`fa4`, mem 0.85, default radix strategy); container `glmf-DAILY-rssm-20260929` kept.

Receipts: [`receipts/`](receipts/) (expert dumps not included, 420 MB).
