# 2026-09-23 — Card E (claim card + long context), GPQA-Diamond at 64K, W3a verify-width

One window, GB300 Station, `lmsysorg/sglang:v0.5.20-cu130` @ `sha256:06e4f2ed21af…`, NVIDIA NVFP4 @ `09b04e5e`, draft `incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd8`, DFlash2 block 7. Fresh container per boot, page cache dropped before each cold boot. Raw artifacts are in `raw/`, with box paths replaced by `$HOME` and the LAN IP by `<station-lan-ip>`. Scrubbed files no longer match the `*SHA256SUMS` pins, which hash the box copies. Unscrubbed and still byte-exact: `test_w3a_unit.py`, `test_w3a2_grammar.py` and `make_w3a_patch.py.gz`. The generator is gzipped only because the repo's credential scanner false-positives on a keyword argument inside it; `gunzip` returns the pinned bytes. `throughput.csv` holds the C1 rows.

## Card E: daily config `e0` (48 KDA slots, mem 0.85, 1M ctx)

| check | result | raw |
|---|---|---|
| cold boot → `/v1/models` | 384 s (warm all shapes +102 s) | `cardE-2026-09-23/e0/boot-time.txt` |
| C1 answer-only (`reasoning_effort: low`), median of 3 | history 196.7 · prose 149.0 · code 284.0 · shell 156.7 tok/s; after the suites 199.7 / 149.3 / 285.0 / 156.7 | `e0/c1-low*.txt` |
| greedy | 20/20 identical to the W2 reference | `e0/greedy-vs-w2a.txt` |
| teacher-forced | 2,857 tokens scored; instrument repeat-identical | `e0/tf-e0.json` |
| tools | 10/10 parsed (twice) | `e0/tool-*.txt` |
| cold prefill (warm shapes, nonce-defeated cache) | 6,825 tok 0.27 s (25.5K tok/s) · 27,437 1.08 s (25.3K) · 54,693 2.03 s (27.0K) · 109,516 4.13 s (26.5K) | `e0/prefill.txt` |
| needle ladder, 3 kinds (single / multi-key / distractor) | **18/18 pass** from 7K to 476K prompt tokens | `e0/needle.jsonl` |
| BFCL dev (600, protocol `harness/protocol.yaml`) | 551/600 = **91.8%** (2 HTTP 400 counted unsolved; the runner prints 551/598 = 92.1%) | `e0/bfcl-e0-dev-summary.json` |
| BFCL held-out (1,311, frozen, run once) | **1,040/1,311 = 79.3%** · unsolved: 184 wrong-arg-value · 31 missing-arg · 29 wrong-function · 18 no-call · 9 HTTP 400 · 0 truncated | `e0/bfcl-e0-heldout.jsonl` |
| BFCL dev, `strict: true` tools (W2) | 516/600 (86.0%), −35 vs non-strict → strict fails dev; per the rule no held-out run | `e0/bfcl-dev-strict.log` |

Held-out cost (protocol formula): mean GPU power 333 W (start/end samples 301/365 W) × 197 s → energy **$0.0026 / 1000 solved** at $0.15/kWh. Amortized Station at $3.80/hr (≈$100k / 3 yr) over 197 s: **$0.20 / 1000 solved**. The 1,311 cases ran at C8, not saturated.

## Card E: `e0b` long-context variant (16 KDA slots, mem 0.90)

`max_total_num_tokens=1,458,304`, 3 running requests. Cold boot 349 s. C1 200.9 / 150.2 / 286.4 / 159.7 (history / prose / code / shell), the same as e0. Needle ladder **9/9 pass** at 476K, 676K and **881K** prompt tokens, all three kinds. Tools 10/10.

## GPQA-Diamond (198 questions, T=0, default = max reasoning effort, C8, shuffled choices)

- At 32K max tokens (Card E): 128/198 = 64.6%, but 69 questions hit the cap.
- The 69 truncated questions were re-asked at **65,536 max tokens** and merged 1:1 by question index (`gpqa64k-2026-09-23/merge_gpqa.py`; the fail-closed checks passed: no missing index, no key mismatch, no errors).
- **Merged: 149/198 = 75.3% (±6.0 pp 95% CI).** 44 questions (22%) are still truncated at 64K and scored wrong. On the 154 questions that finished, accuracy is 149/154 (96.8%). Finished answers used a median of 2,244 completion tokens (max 50,841). The 25 that only finished in the 64K rerun used a median of 14,345.
- Read this as a **lower bound at a 64K cap**, not as the model's GPQA. We ran it, but it is a sanity anchor and not a leaderboard entry.

## W3a: fixed verify width (drafter block 7, target verifies the first K)

The patch replaces two files in the image: `speculative/dflash_worker_v2.py` and `layers/attention/dsa/kpool_plan.py`. The drafter still proposes 7 tokens, and the target verifies only the first K. The code and diffs are in `raw/w3a2-2026-09-23/`.

1. **W3a (invalid).** Every gate passed, but the verify-width counter read `steps=0 fallback_full_width=4097`, so the patch never ran. The card logged that counter but did not fail on it.
2. **W3a.1 diagnostic.** Every fallback had `why=grammar`. With `--tool-call-parser glm47` and non-strict tools, SGLang's `serving_chat.py` attaches a `full_assistant_ebnf` grammar to **every chat request**, tools or not. The patch excluded grammar steps, so it excluded all real traffic.
3. **W3a.2.** The grammar tree is built over the same K-wide chain that the target verifies. A new unit test (`test_w3a2_grammar.py`) uses the real GLM grammar and tokenizer. It checks that the K-chain mask equals the first K rows of the 7-chain mask for K = 5, 4, 3, 2, plain and reasoning-wrapped: 320 comparisons, 0 mismatches. The original 28 unit tests also pass. The boot now **fails** unless the counter shows steps > 0 and fallback < 5%. Measured: 100% narrow on 4,865 / 5,121 / 5,889 steps (K = 5 / 4 / 3).

Results against the adjacent stock control `ctrl-e0` (median of 3, tok/s):

| | history | prose | code | shell | accept len | greedy vs e0 | TF vs e0 | tools |
|---|---|---|---|---|---|---|---|---|
| ctrl-e0 | 200.4 | 149.9 | 286.1 | 158.8 | 3.40 | 20/20 | — | 10/10 |
| K=5 | 208.4 (+4.0%) | 160.4 (+7.0%) | 277.4 (−3.0%) | 165.4 (+4.2%) | 3.05 | **20/20** | Δ 0.0 | 10/10 |
| K=4 | 210.0 (+4.8%) | 165.6 (+10.5%) | 264.5 (−7.6%) | 153.8 (−3.2%) | 2.77 | **20/20** | Δ 0.0 | 10/10 |
| K=3 | 200.7 (+0.2%) | 176.2 (+17.6%) | 235.1 (−17.8%) | 150.0 (−5.5%) | 2.38 | **1/20 ✗** | Δ 0.0 | 10/10 |

**Verdict: no width wins, so W3a is parked.** The win bar was prose and essay up with code no worse than −1%. Narrower verification helps the classes where the drafter's acceptance is low (prose) and hurts the ones where it is high (code). Any fixed K trades one class for the other.
- K=5 and K=4 are lossless against stock: greedy 20/20, teacher-forced Δ 0.0.
- K=3 diverges on greedy (1/20) while the teacher-forced logprobs of the stock text are unchanged. The divergence is therefore in the narrow-verify decode path, not the model. The cause is not yet diagnosed.
- The teacher-forced harness prints `NONINFERIOR=False` only because this prompt set has 2,857 tokens and the harness requires ≥3,000. The max Δ is 0.0.
- The adaptive switch (W3b: narrow when recent acceptance is low, full when it is high) was conditional on W3a passing. It has not been built.
