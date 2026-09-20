# 2026-09-20 — Fidelity: teacher-forced Δlogprob and GPQA-Diamond, run by us

The recipe card carried two amber cells since #36: **Δlogprob vs a no-hook reference** and **a GPQA anchor we ran ourselves**.
This bundle fills both. It also carries the v19 promotion receipts' fidelity side: the captures below were taken on the exact
containers that became `dsv41-vllm-v19-…-BOUND-REF` (12:11 CDT) and `dsv41-vllm-v18-cgsizes-RETIRED-REF`.

Worker: Milo (Hermes `milo` profile, claude-fable-5-1 via anthropic). Runner: `fidelity-runner-2026-09-20.sh` (nohup on the
box, chained after the v19d window). Approved and read by James Meadlock.

## 1. Teacher-forced Δlogprob — `tf_logprob.py`

**Mechanism.** `/v1/completions` with `max_tokens=1, prompt_logprobs=1, temperature=0`. vLLM returns, for every prompt
position, the logprob of the *actual* next token and the top-1 token. This is prefill only — the DSpark drafter and the
verify/accept loop never run — so the number isolates the serving path's numerics (offload layout, hook finalize, KV dtype,
kernels) from speculation. Corpus (`tf_corpus.jsonl`, built by `build_tf_corpus.py`): the 18 e5 parity prompts each followed
by their v18 greedy output, plus 3 × 14k-char slices from each of the 7 fund-harness 10-Ks. **39 docs, 79,544 positions**,
capped at 4,096 tokens per doc. Tokenization matched on all 39 docs for every pair (a mismatch would void the doc).

**Reference.** `dsv41-vllm-v14-1M-ksched-agent-RETIRED-REF`: the same 0909 image as v18, k-schedule, `--cpu-offload-gb 60`,
**no hook**. So `ref → v18` measures what the pin-hot-experts hook does to the logits; `ref → v19` measures hook + nightly image
(`dee37d89`) + `fp8_ds_mla` KV + off54 together.

| pair | positions | mean \|Δlp\| | p50 | p99 | max | top-1 flips | ppl ref → cand |
|---|--:|--:|--:|--:|--:|--:|--:|
| no-hook → v18 | 79,544 | **0.0300** | 0.00001 | 0.655 | 9.63 | 841 (**1.06%**) | 1.5077 → 1.5085 |
| no-hook → v19 | 79,544 | **0.0592** | 0.00008 | 1.109 | 11.9 | 1,736 (**2.18%**) | 1.5077 → 1.5070 |
| v18 → v19 | 79,544 | 0.0603 | 0.00008 | 1.110 | 11.9 | 1,791 (2.25%) | 1.5085 → 1.5070 |

**Read.** Median Δ is 1e-5 (v18) / 8e-5 (v19) — accumulation-order noise from the hook's two-call fp32 finalize and, for v19,
the fp8 KV path and nightly kernels. The tail is real: ~1% (v18) / ~2% (v19) of positions change argmax and the worst
positions move by ~10 nats. Perplexity over the corpus is unchanged to four decimals in every pair, and v19's is marginally
*lower* than the no-hook reference, so this is reordering, not degradation. **It is not bit-exact and the card will not say
"pass"; it says the number.** Greedy-output parity (44/44 non-tool exact on v15 e2c; 18/18 v19a-vs-v19b today) is consistent
with this: greedy decoding hides sub-argmax movement, teacher forcing does not.

Per-doc breakdown is in `receipts/tf-compare-*.json` (`per_doc.<id>.{mean_abs_dlp,max_abs_dlp,top1_flips}`). The agent-prompt
docs (short, tool-shaped) carry most of the flips; the 10-K slices are quieter.

Runner note: the first v19 capture (`tf-v19c.partial.*`) died at doc 20 on one transient HTTP 400 while the runner had the
v19c container; `tf_logprob.py` now retries once and records per-doc errors instead of aborting. The full `tf-v19.jsonl`
was re-captured at 12:46 CDT on the promoted v19 container — same image, same weights, same hook, same autotune set; the
partial 8k-position compare (`tf-compare-*-vs-v19c.json`) is kept for provenance and agrees in kind (mean 0.12 on the
flip-heavy agent docs only).

## 2. GPQA-Diamond (198) — `gpqa_diamond.py`

Dataset `Idavidrein/gpqa` `gpqa_diamond.csv` (gated; James accepted the terms; the CSV is **not** committed). Fixed choice
shuffle (seed 0 per question), answer required as `Answer: X` after the reasoning, letter-extraction grader, **no-parse =
wrong**. `temperature 0`, reasoning on, C16, no system prompt.

| run | max_tokens | correct | acc | 95% CI | no-parse / truncated | reasoning tokens mean / p95 | wall |
|---|--:|--:|--:|--:|--:|--:|--:|
| v18 (0909 + hook, `-BOUND-REF` at the time) | 16,384 | 159/198 | **80.3%** | ±5.5 | 24 / 24 | 4,286 / 16,384 (cap) | 25.7 min |
| v19 (promoted reference) | 65,536 | 173/198 | **87.4%** | ±4.6 | 6 / 6 | 6,252 / 24,652 | 39.5 min |

All 24 v18 misses-by-parse are truncations at the 16k cap (`truncated=24`, same indices) — the model was still reasoning.
The v19 run raises the cap to 64k so the number reflects answers, not the cap.

Model card: 90.9 at *max* reasoning effort, T=1.0. Ours is T=0 with the default effort and a plain "Answer: X" format —
**a sanity anchor that the served weights reason at the expected level, not a reproduction of the card**. Do not compare the
two as if they were the same protocol.

## Files

- `gpqa_diamond.py`, `tf_logprob.py`, `build_tf_corpus.py`, `fidelity-runner-2026-09-20.sh` — instruments (also in
  `pin-hot-experts/scripts/`).
- `tf_corpus.jsonl` — the exact teacher-forcing corpus (LAN hosts scrubbed).
- `receipts/tf-{v14nohook,v18,v19}.jsonl` — per-position tokens / logprobs / top-1 (~2.6 MB each); `tf-compare-*.json`.
- `receipts/gpqa-v18.jsonl`, `receipts/gpqa-v19.jsonl` — per-question gold / pred / tokens / finish reason; `gpqa-*.log`.
