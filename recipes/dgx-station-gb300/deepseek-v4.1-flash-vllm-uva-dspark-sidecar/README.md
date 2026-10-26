# DeepSeek-V4.1-Flash · 1M context on a DGX Station GB300 + RTX PRO 6000 sidecar

The 510 GB DeepSeek-V4.1-Flash-1017 checkpoint does not fit one GB300. The GB300-only recipe in this repo
solves that by offloading 60 GiB of routed experts to pinned Grace memory over NVLink-C2C — which works, and
leaves a serial ~350 GB/s read on the critical path of every decode step. This recipe takes the other road:
keep the hot experts in HBM and move the cold ones to the **RTX PRO 6000 Max-Q in the same chassis, where
they are computed**, not read. Only activations cross PCIe.

Same box, same checkpoint, same harness, bracketed controls in one window (2026-10-22, measured on the v25 base
launch; the serving release v27 adds the v25m map, a Grace prefix-KV tier and block verification, whose deltas are
Cards GV, KV3 and BR1 below):

| | GB300 only (sibling recipe, v23) | GB300 + RTX 6000 sidecar (this recipe) |
|---|--:|--:|
| Decode C1, real text 8.2K/1,024 | 187.5 tok/s | **279.5 tok/s** (+49%) |
| Decode C16 | 524.5 | **1,576.9** (3.0×) |
| Decode C32 | 936.9 (max-num-seqs 24, queues) | **2,400.3** (2.6×) |
| TTFT, 8.2K prompt | 0.66 s | **0.19 s** |
| Cold 64K prefill | 23,439 tok/s | **50,119 tok/s** |
| 1M-token prompt, first token | 81.4 s (12,262 tok/s) | **33.2 s** (30,069 tok/s) |
| KV pool at 1M ctx | 2,504,891 tokens (util 0.97) | **4,400,013 tokens** (`--kv-cache-memory` 8.5 GiB; 2,537,081 at util 0.97) |
| Host memory pinned | ~253 GiB (Engram + cold experts) | Engram only — cold experts are on the card |
| Extra process you must keep alive | — | one sidecar container + a fail-closed watchdog |
| Wins on | simplicity, no second GPU, prefill is throttled the same | decode at every concurrency, TTFT, long-context reads |

Both recipes serve the same weights and stay published; this one is the faster lane on this hardware, the
sibling is the one that needs nothing but the GB300.

> **Lineage, stated up front.** The production shape of this recipe — two-step `sidecar` → `server` launch, the
> b12x peer tier-2 protocol, the `rowmap-mix-v1` calibration, the published same-chassis numbers, and the
> `kbench_chat_ds` harness this recipe is measured on — is **RonanLabsAI's**
> ([dgx-station-frontier](https://github.com/RonanLabsAI/dgx-station-frontier),
> `recipes/dsv41-flash-sidecar`, Apache-2.0), built on **original-el8's** hook. What is ours is the delta:
> the 295/89 map, the agreement gate run on real prompts, the ack-before-bookkeeping sidecar fix, the
> fail-closed watchdog and the bracketed measurement discipline. See [Credits](#credits).

> **Status, 2026-10-26 16:45 CDT (recipe stays experimental): v27 is the serving release.** Lane `:30006` on the
> reference box serves **v27 = `dsv41-vllm-1017-v27-block-BOUND-REF` + `-rtx`**: the v25m-kvtier sidecar launch
> (`ROWMAP=rowmap-mixed-ar-v1.json`,
> `EXTRA="--kv-cache-memory 9126805504 --kv-offloading-size 48 --kv-offloading-backend native"`) plus one spec key,
> `{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,32,3]],"draft_sample_method":"probabilistic","rejection_sample_method":"block"}`,
> image nightly-af7f9488 (`sha256:3af6a45b…`), KV pool 4,400,013 tokens, fail-stop watchdog in enforce mode. It was
> promoted on 2026-10-26 04:08 CDT on Card BR1 (PASS re-scored, James's call): hash-hit boot 528 s, smoke
> (`reasoning_effort: low`, tool call) and consumer checks (Hermes one-shot with tool calls, Forge guest gateway
> `deepseek-v4.1-flash`) passed; promotion log `results/2026-10-26-br1-block-verification/raw/promote-v27.log`.
> Rollback pair `dsv41-vllm-1017-v25m-kvtier-RETIRED-REF` (+`-rtx`, v27 without the block key), stopped-and-kept.
> During the day the same box ran four GB300-only sibling cards and one park window. v27 was stopped for each and
> restored after each (after BR3 06:49, BR4 10:28, park 11:06, BR5 13:52 and BR6 16:36 CDT). Every restore was
> re-verified: KV 4,400,013, the block key in the args, and the watchdog re-armed with PPID 1. The Forge guest
> gateway re-listed the lane at 16:45 CDT. Earlier history: from 2026-10-24 23:16 to 2026-10-26 03:58 CDT the lane
> served `dsv41-vllm-1017-v25m-kvtier` + `-rtx` (the same launch without the key). *(Corrected 2026-10-26: an
> earlier banner said the lane served the GB300-only sibling's v26 config. That was true only from 2026-10-24
> ~21:30 to 23:16 CDT.)* The adaptive-residency campaign has ended, and the v25m 48 h soak was cancelled on
> 2026-10-24, so no 48 h soak has run on v27 either.
> **Cross-links:** GB300-only sibling
> [`../deepseek-v4.1-flash-vllm-uva-dspark`](../deepseek-v4.1-flash-vllm-uva-dspark/README.md). Blogs:
> [GB300 + sidecar vs 4× RTX PRO 6000 (BRRR)](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html),
> [Adaptive Residency Part 3, the seed map](https://al-engr.com/gb300-adaptive-residency-seed-map.html) and
> [Part 1, with its convergence correction](https://al-engr.com/gb300-adaptive-expert-residency.html). Peer recipe:
> [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR).

## 2026-10-26 (pm): what happened on the GB300-only sibling, and why it does not change this recipe

The GB300-only sibling ([`../deepseek-v4.1-flash-vllm-uva-dspark`](../deepseek-v4.1-flash-vllm-uva-dspark/README.md)) tried the same key on its own lane: no sidecar, Grace
offload, a **greedy** drafter and the k schedule `[[1,8,5],[9,24,1]]`.

- **Block verification alone was a null there** (Card BR3,
  [`results/2026-10-26-br3-block-verification-null/`](../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-26-br3-block-verification-null/)): tokens per verify step 0.999× the control. Block
  verification works on the draft distribution. A greedy drafter gives it a point mass, and it degenerates to
  standard verification.
- **Probabilistic drafting was the lever** (Card BR4,
  [`results/2026-10-26-br4-probabilistic-drafting/`](../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-26-br4-probabilistic-drafting/)): +10.3% tokens per step at temperature 1.0. It is now
  that recipe's reference, `v28-prob`.
- **Block on top of probabilistic drafting never passed a bracket cleanly in three tries:** BR4's PB arm, BR5
  ([`results/2026-10-26-br5-block-rebracket/`](../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-26-br5-block-rebracket/)) and BR6 with a redesigned G6
  ([`results/2026-10-26-br6-block-g6-redesign/`](../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-26-br6-block-g6-redesign/), added by a parallel PR). Its speed and teacher-forced gates
  passed, but a temperature-0 parity gate tripped each time (shell_ops in BR4's G2b; tool-turn emission in G6 on
  BR4 and BR5; BR6 in its own bundle). The sibling retired the key for cause.

**Why this lane is different:** its drafter was already probabilistic. `"draft_sample_method":"probabilistic"` has
been in this launch since v25, so block verification had a distribution to work on. Card BR1 passed here: +2.0%
tokens per step, no class lost beyond noise (C1 0.996–1.054×, C8 1.008–1.082×), TF flips at the control floor,
gauntlet 12/12 and replay_turns inside the controls. The pass is the re-score; the pre-registered FAIL was two
instrument defects, and both verdicts are kept (see below). v27 stays the serving release. None of the sibling's
cards were measured on this lane. Write-up with the addendum:
[GB300 + sidecar vs 4× RTX PRO 6000](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html).

## 2026-10-26: block verification (Card BR1) — PASS, PROMOTED as v27; k ladder (Card BR2) — k=9 loses; peer harness side-by-side

### Card BR1 — block verification, one key

```bash
# production spec config + one key (rejection_sample_method), everything else unchanged
SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,32,3]],"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

Block verification ([Sun et al.](https://arxiv.org/abs/2403.10444)) verifies the drafted block jointly instead of
token by token. The output distribution is the same. **Result: PASS (re-scored); promoted to `:30006` as v27 on
2026-10-26 04:08 CDT** (James's call; promotion log `results/2026-10-26-br1-block-verification/raw/promote-v27.log`).

Bracketed A (control) → B (block) → A2 (control) on the then-production v25m-kvtier lane config, 2026-10-25
19:33–21:35 CDT.
All three boots were autotune hash hits (528–529 s; `rejection_sample_method` is not a compute-hash factor). Every
arm had KV 4,400,013, and the block key appeared in the args on B only (0/1/0, read from `docker inspect`). Every
arm read 0 foreign clients (the lane was firewalled to loopback while dark), 0 sidecar timeouts, 0 watchdog trips
and 0 FF errors. vLLM af7f9488 applies block verification **only on non-greedy rows** (`USE_BLOCK_VERIFICATION and not
is_greedy`). The primary window was therefore temperature 1.0 / top_p 1.0, seeded, with 40 real transcripts per
class and 256 forced tokens, r1/r2. The stock temperature-0 window was kept as a parity gate. Receipts:
[`results/2026-10-26-br1-block-verification/`](results/2026-10-26-br1-block-verification/). Card:
[`card-br1.png`](card-br1.png).

| temp 1.0 decode tok/s | C1 A / A2 / **B** | B ÷ ctrl | C8 A / A2 / **B** | B ÷ ctrl |
|---|---|--:|---|--:|
| prose | 266.7 / 270.2 / **278.3** | 1.037 | 972.4 / 978.7 / **983.0** | 1.008 |
| structured | 302.2 / 300.8 / **305.3** | 1.013 | 1,059.8 / 1,024.8 / **1,068.2** | 1.025 |
| code | 352.8 / 353.0 / **357.3** | 1.013 | 1,173.9 / 1,159.6 / **1,221.1** | 1.047 |
| shell_ops | 374.0 / 385.5 / **378.1** | 0.996 | 1,194.2 / 1,245.8 / **1,320.3** | 1.082 |
| tool_json | 333.4 / 326.4 / **347.6** | 1.054 | 1,184.1 / 1,156.0 / **1,202.5** | 1.028 |

- **Controls:** A and A2 agree at C1 within 0.1–3.0% per class.
- **Acceptance:** tokens per verify step (1 + accepted/drafts, `/metrics` deltas over the temp-1.0 window) A 3.204,
  **B 3.262**, A2 3.189 (B/ctrl 1.020). Mixed C1: 2.092 / 2.105 / 2.088.
- **Mixed traffic:** `ar_mixed_t`, C1, natural length, ~65 prompts per 120 s rep, temp 1.0. A 212.05, **B 212.77**,
  A2 212.65 (+0.20%).
- **Temp-0 window:** B is within noise on every class (prose −0.9%, structured −0.2%, code −2.2%, shell +1.1%,
  tool_json +0.7%), against bars of 2.6–4.1% that include the controls' own r1/r2 spread.
- **Fidelity:** teacher-forced top-1 flips B-vs-A 2.166% and B-vs-A2 2.176%, against an A2-vs-A floor of 2.172%
  (79,544 positions, 39 docs, 0 token-mismatch docs). Margin-weighted flips: hi_agent_tool = 0 on all three
  compares, W 2.839 / 2.751 against a floor of 2.773. Gauntlet 12/12 on A, B and A2. replay_turns tool-turn emission
  B [7, 7]/12 vs A [6, 8] and A2 [7, 6]. The x3 acceptance canary (temp 0) is within noise. The production watchdog
  was re-armed after restore with PPID 1.

### Card BR2 — k ladder (A → K9 → A2) + the peer harness on every arm

K9 = block verification + `num_speculative_tokens 9` + `[[1,4,9],[5,8,3],[9,32,3]]`. The controls A and A2 are the
v25m-kvtier config, which was production when BR2 ran, **without** block verification. The K9 ratios therefore
measure block + k=9 against that config, not against v27. *(Corrected 2026-10-26: this said "the production
config"; production is now v27, which has block verification on.)*
Run 2026-10-25 21:36–23:24 CDT; receipts:
[`results/2026-10-26-br2-k-ladder-peerbench/`](results/2026-10-26-br2-k-ladder-peerbench/).

**K9 booted as a hash hit (529 s, Loaded 117 / Saved 117, 0 new)**: `num_speculative_tokens` is not in the autotune
hash on af7f9488. The checkpoint's `dspark_block_size` is 5. The DSpark speculator emits N =
`num_speculative_tokens` query positions (the anchor plus N−1 noise positions) in one pass, and the engine does
not cap N. CUDA-graph tokens 32×(k+1) come to 320 on K9 and 192 on A, both above the 128-entry capture list.
Production is already above 128, but the batch schedule caps k at 3 above batch 4, so the largest real decode batch
is 128 on every arm. KV was 4,400,013 on all arms.

| K9 ÷ mean(A, A2), temp 1.0 | prose | structured | code | shell_ops | tool_json | mixed |
|---|--:|--:|--:|--:|--:|--:|
| C1 | 0.813 | 0.860 | 0.868 | 0.918 | 0.853 | 0.793 (168.4 vs 212.3 / 212.5) |
| C8 | 0.896 | 0.891 | 0.925 | 0.930 | 0.882 | — |
| tokens/step C1 · A (k5) | 2.85 | 3.03 | 3.83 | 3.98 | 3.63 | 2.08 |
| tokens/step C1 · K9 (k9) | 2.60 | 2.94 | 3.66 | 4.04 | 3.45 | 1.82 |
| tokens/step C1 · A2 (k5) | 2.91 | 3.15 | 3.80 | 3.97 | 3.66 | 2.09 |

Per-position acceptance at C1, temp 1.0: curve[p] = accepted-at-p / drafts, marginal = curve[p] / curve[p−1].

| class · arm | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prose · A k5 | .717 | .476 | .310 | .205 | .145 | | | | |
| prose · K9 | .612 | .379 | .235 | .147 | .097 | .063 | .036 | .020 | .011 |
| prose · K9 marginal | .61 | .62 | .62 | .62 | .66 | .65 | .57 | .57 | .54 |
| structured · K9 marginal | .66 | .69 | .67 | .67 | .71 | .65 | .55 | .53 | .55 |
| code · A k5 | .823 | .653 | .536 | .445 | .371 | | | | |
| code · K9 | .719 | .548 | .428 | .322 | .255 | .179 | .108 | .064 | .037 |
| code · K9 marginal | .72 | .76 | .78 | .75 | .79 | .70 | .60 | .60 | .57 |
| shell_ops · K9 marginal | .76 | .81 | .80 | .78 | .82 | .72 | .63 | .53 | .51 |
| tool_json · K9 marginal | .70 | .73 | .75 | .76 | .78 | .68 | .62 | .56 | .67 |

**Reading:** marginal acceptance holds at 0.61–0.82 through position 5 and drops to 0.51–0.67 from position 7.
The drafter is extrapolating past its trained block there. Position-1 acceptance also falls under the longer
block (prose .72 → .61, code .82 → .72). **k=5 stays.** All curves, the C8 rows and A2's are in
[`raw/per-position-curves.json`](results/2026-10-26-br2-k-ladder-peerbench/raw/per-position-curves.json). The
per-position histogram (`vllm:spec_decode_num_accepted_tokens_per_pos`) is now a standing instrument.

### Peer harness side-by-side

[`scripts/peerbench.py`](results/2026-10-26-br2-k-ladder-peerbench/scripts/peerbench.py) is a stdlib port of
[fraserprice's BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) `bench.py` (Apache-2.0, credit
header kept). It mirrors their method: a coding prompt padded to 2K/16K/64K through `/tokenize` with a nonce per
stream. Each round makes two passes: `max_tokens 1` for TTFT/PP, then the same prompts from cache so all streams
decode together. TG counts only while every stream decodes, with natural EOS ≤ 2048, thinking off, and temperature
1.0 / top_p 1.0 as their bench sends. Ours ran **2 rounds (theirs 5)** with no cool-down wait (the hottest GPU is
logged). Our server omits `prompt_tokens_details`, so the cache-bust check reads the `/metrics` prefix-cache counter.

| prompt · conc | TG p50 | PP p50 tok/s | TTFT p50 | TG total | accept |
|---|---|---|---|---|---|
| | **theirs / ours** | **theirs / ours** | **theirs / ours** | **theirs / ours** | **theirs / ours** |
| 2K · C1 | 703 / 466 | 12,452 / 14,279 | 172 / 152 ms | 706 / 467 | 5.94 (71%) / 5.13 (83%) |
| 2K · C4 | 329 / 321 | 4,470 / 8,137 | 489 / 264 ms | 1,320 / 1,287 | 5.94 / 5.15 |
| 2K · C8 | 231 / 213 | 2,589 / 5,259 | 829 / 408 ms | 1,847 / 1,697 | 5.97 / 3.76 |
| 2K · C16 | 166 / 169 | 1,610 / 3,357 | 1,333 / 643 ms | 2,648 / 2,719 | 5.96 / 3.70 |
| 16K · C1 | 684 / 471 | 16,953 / 47,297 | 972 / 349 ms | 691 / 470 | 5.99 / 5.21 |
| 16K · C4 | 321 / 319 | 7,199 / 16,356 | 2,362 / 1,025 ms | 1,281 / 1,277 | 5.95 / 5.05 |
| 16K · C8 | 226 / 208 | 4,072 / 10,271 | 4,089 / 1,617 ms | 1,806 / 1,665 | 5.99 / 3.76 |
| 16K · C16 | 158 / 170 | 2,167 / 5,944 | 7,624 / 2,781 ms | 2,522 / 2,715 | 5.98 / 3.70 |
| 64K · C1 | 675 / 453 | 17,852 / 46,246 | 3,677 / 1,419 ms | 669 / 454 | 5.97 / 5.05 |
| 64K · C4 | 312 / 316 | 7,435 / 18,847 | 9,165 / 3,597 ms | 1,243 / 1,266 | 6.02 / 5.13 |
| 64K · C8 | 210 / 209 | 4,051 / 10,860 | 16,396 / 6,098 ms | 1,686 / 1,675 | 5.95 / 3.81 |
| 64K · C16 | 131 / 165 | 2,131 / 6,045 | 30,893 / 10,890 ms | 2,096 / 2,653 | 5.98 / 3.72 |

**Theirs** is the "Thinking off" table published on the BRRR model card, copied and not re-run by us: 4× RTX PRO
6000 Blackwell Max-Q at 325 W, Threadripper 9960X, 64 GB RAM, Samsung 990 PRO, adaptive verification off, DSpark 7
draft tokens, 5 rounds. **Ours** is Card BR2 arm A: one GB300 plus one RTX PRO 6000 Max-Q sidecar, this lane's
v25m-kvtier config (production until v27; no block verification), DSpark k=5 at C1–C4 and k=3 above batch 4 (hence our lower C8/C16 accept), 2 rounds.
[`raw/A/peerbench-BR2-A.md`](results/2026-10-26-br2-k-ladder-peerbench/raw/A/peerbench-BR2-A.md). Arm A2 agrees
with A within 0–2.5% on TG in all 12 cells. Their box wins single-stream decode by 1.45–1.51×. Ours wins prefill
2.3–2.8× at 16K/64K, TTFT at every cell, and TG at 64K/C16. At C4–C16 the two are mostly level.

**Decomposition, 2K/C1.** Theirs: 703 = 5.94 tok/step × 118 steps/s (8.45 ms per step). Ours: 466 = 5.13 × 91 (11.0
ms). The 1.51× gap = 1.16× tokens per step × 1.30× steps per second. Tokens per step favour them because k=7 on a
friendly codegen prompt beats our k=5. Step time is the bigger factor. Two hypotheses, not tested in this card: our
sidecar hop grows with the expert union under high acceptance, and their persistent fused `decoder.cu` removes
per-step small-kernel churn. The `(71%)` in their accept column is accepted ÷ draft tokens = 4.94 / 7, not an
independent per-position rate. On the same harness **K9** ties A at 2K/C1 (468, 5.70 tok/step), wins 64K/C1 (489 vs
453, 6.05 tok/step), reads 316 vs 319 at 16K/C4 (5.95), and loses 4–7% at every C8/C16 cell. Its step time at 2K/C1
is 12.2 ms. So k=9 pays only on padded single-stream codegen, and it loses on real traffic. Write-up:
[GB300 + sidecar vs 4× RTX PRO 6000](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html).

### Instrumentation note — both verdicts kept

Both cards' pre-registered verdicts failed on instrument defects, not on measurements. **BR1: FAIL
`BR1FAIL-G0-G2b`.** BR2: INVALID-CONTROL-A-A2. G0 compared the container env as an ordered list; Docker returns the
same 44 variables in varying order, and the sets were identical on every arm. In BR1, G2b's temperature-0 no-op bar
was tighter than the class's own r1/r2 spread (code/C1 read −2.2% against a ±2% bar, with a 2.1% control r1/r2
spread). The re-scores (`br1_verdict_rescore.py`, `br2_verdict_rescore.py`) compare env as sets and widen the G2b
bar to honour the controls' r1/r2 spread. They read **PASS `BR1PASS`** and **PASS `BR2PASS`**; no speed, fidelity or
watchdog gate moved. Each bundle keeps `verdict.json` / `VERDICT.txt` as pre-registered and `verdict-rescore.json` /
`VERDICT-rescore.txt` as re-scored.

## 2026-10-24: kv-host-tier variant, a 48 GiB Grace prefix-KV tier (Card KV3)

**Result: full battery PASS, 10/10 pre-registered gates, no measurable speed cost, eligible for production use as a variant.**
The change is two flags added to the existing launch, after the KV declaration already published:

```bash
EXTRA="--kv-cache-memory 9126805504 --kv-offloading-size 48 --kv-offloading-backend native" bash scripts/launch-sidecar.sh server
```

With the tier on, vLLM's `OffloadingConnector` keeps prefix blocks evicted from the GPU in a 51.54 GB `/dev/shm`
mmap in Grace memory. When a later request hits that prefix, it loads the blocks back instead of recomputing
them. The GPU pool stays at **4,400,013 tokens on every arm**, so the tier only adds capacity. In the evict-replay
probes the tier loaded **1,656 B per hit token**, so 48 GiB holds **~31M prefix tokens** on top of the GPU's 4.40M.
That figure was measured on this lane's `fp8_ds_mla` KV and does not transfer to other KV formats.

Bracketed **K0 (no tier) → K1 (tier) → K0R (no tier)**. Each arm got a fresh sidecar+server boot on the v25m map,
with the fail-closed watchdog armed, on 2026-10-24 between 21:22 and 22:35 CDT. Receipts:
[`results/2026-10-24-kv3-host-prefix-tier/`](results/2026-10-24-kv3-host-prefix-tier/).

| cell | no tier (K0 / K0R) | **tier (K1)** | Δ / gate |
|---|--:|--:|---|
| C1 held-out mixed JP/DE + agent, 6 × 120 s, mean | 219.76 / 219.20 | **219.89** | +0.19% vs pooled 219.48 (bar −2.04%; drift K0→K0R 0.26%) |
| cold 64K TTFT proxy, median | 1.401 / 1.406 s | 1.424 s | +1.47% (bar +5%) |
| warm replay r2 (8,172-tok prompt, 256 forced) | 195.52 / 183.72 | 177.48 | −3.40% vs the lower control (bar −5%); the controls themselves span 183.7–195.5 |
| evict → `/reset_prefix_cache` → replay, 7,929 / 31,629 tok | — | host-tier hit + load, **identical output**, ×1.54 / ×3.2 | PASS |
| 954,028-token needle | recalled, 32.9 / 33.0 s | recalled, 32.8 s | PASS |
| GPU KV tokens | 4,400,013 | 4,400,013 | unchanged |
| MemAvailable min during cells | 177.7 / 178.9 GiB | 129.8 GiB | floor 60; 0 new OOM kills |
| watchdog timeouts / trips · error lines · pin warnings | 0 / 0 · 0 · — | 0 / 0 · 0 · 0 | PASS |
| gauntlet | — | 12/12 | PASS |

- **The card's launcher (`launch_kv3.sh`) set `VLLM_SERVER_DEV_MODE=1`. That was for the smoke and probe only:** it
  exposes `/reset_prefix_cache`, which the evict-replay probe needs to drop the GPU prefix cache. The production
  variant is the two flags above, without dev mode. That exact no-dev-mode boot was not measured separately.
- **Upstream [vllm#47890](https://github.com/vllm-project/vllm/issues/47890)** reports that speculative decoding can
  zero CPU-tier hits. We tested for it, and it did not reproduce on this image (af7f9488): with DSpark on, both
  probe lengths (7,929 and 31,629 tokens, neither a multiple of 256) hit and loaded from the host tier. The K1 boot
  logs `treating all groups as non-draft for offloading`. Only two lengths were probed.
- **LMCache is not an option on this lane.** It needs the hybrid KV manager turned off.
- **Limits:** the base was the v25m map, not the static map. Fidelity evidence on the tier arm is identical short
  outputs on the probes plus gauntlet, needle and zero errors. No teacher-forced, GSM8K or BFCL run was done on the tier arm.
  Hit rates under real multi-user eviction pressure, and C>1 with the tier, were not measured. The tier also costs
  about 48 GiB of host MemAvailable. Card: [`card-kv.png`](card-kv.png) ([`card-kv.yaml`](card-kv.yaml)).

## 2026-10-23 — experimental variant v25m: a mixed-traffic map (no code change)

The 295/89 split in `rowmap-static-v1.json` was fitted on agent traffic. On Japanese/German prose, the GB300 spent
**11.8 % of wall time waiting on the RTX** (0.5 % on agent English, measured from the peer shm stall counters).
v25m changes **one file**: `rowmap-mixed-ar-v1.json`, the top-295 experts per layer by cumulative decode counts
collected on ~100k JP/DE and ~105k agent/tool tokens during the
[GB300-only adaptive-residency campaign](../deepseek-v4.1-flash-vllm-uva-dspark/README.md). Same image, hook, b12x, flags and KV.

| post-60 s medians, tok/s | v25 (static map) | **v25m (mixed map)** |
|---|---:|---:|
| mixed JP/DE + agent, C1 aggregate, **held-out prompts** | 208.9 | **221.1** (+5.8 %) |
| — per request, JP/DE / agent | 179.5 / 250.8 | **206.7** / 246.4 |
| mixed, C4 aggregate, held-out | 560.4 | 558.5 (no change) |
| shift protocol JP/DE / agent, C1 (in-sample prompts) | 186.0 / 244.3 | **206.9** / 247.9 |
| GB300 stall on the RTX, JP/DE / agent | 11.8 % / 0.5 % | **0.5 %** / 0.4 % |

- **What it buys:** single-stream off-domain prose, +11–15 %. At C4 the GB300's own step time already hides the RTX work.
- **Fidelity (Card GV, 2026-10-24): 9 of 10 gates pass, the bring-up cosine gate fails.** Greedy parity cannot gate this lane (the same map self-agrees 14/50), so the gates are SC2's. Teacher-forced flips PASS against both v23 and v25 static; GSM8K-1319 96.82 vs 96.97; BFCL dev/held-out **identical** to v25 (556/600, 1018/1311); gauntlet 12/12; needle 500/500; tool emission, effort and image probes pass; 0 sidecar timeouts. **The bring-up cosine gate misses: min cos 0.99811 vs the 0.999 bar.** That is one of 1,439 gated checks, at layer 39 (the worst layer on every map), and the rest of the distribution matches v25 static (p1 0.99968 vs 0.99970). The same gate rejected the 285/99 map, so the bar stays where it is and v25m stays **experimental**. **Follow-up (Card GV2):** restoring layer 39 to v25's exact static split did not fix it (min 0.99831; worst check again layer 39, where the cold experts carried ~0 % of the output norm); see [`results/2026-10-24-gv2-l39-static/`](results/2026-10-24-gv2-l39-static/). **Correction (2026-10-24 08:15):** an earlier version said the v25m miss was check noise, not the map. That overreached. GV2 shows layer 39's checks are noisy when the cold experts carry ~0 of the output, but v25m's own failing check had them at 7.9 %, so GV2 does not clear v25m. The open question goes to a new, pre-registered gate (R2, below). **Cosine gate R2, registered 2026-10-24 before re-judging** ([`results/2026-10-26-gs-cosine-r2/`](results/2026-10-26-gs-cosine-r2/)): judge the sidecar's error as a fraction of the whole layer output, e = rel_diff × cold share, against a bar of 1.5 × the worst e from three v25-static check boots (the SC0 boot plus two more after the soak, 2026-10-26). Every archived boot is re-judged and the table published whatever it says. *(Corrected 2026-10-26: the re-judge
has not run. Card GS was cancelled with the soak on 2026-10-24, and v27 serves on this map with the miss open; see
Known limits.)* Receipts: [`results/2026-10-24-gv-v25m-gates/`](results/2026-10-24-gv-v25m-gates/).
- **Same-instrument speed vs v25:** real-text English C1 273.2 vs 279.2 (−2 %), knee C8 887 vs 934 (−5 %), C16 1,411 vs 1,409. The mixed map trades a little English-only speed for +15 % on JP/DE.
- **Soak:** 48 h on v25m, restarted after Card GV2 at 07:12 Sat 10-24 → 07:12 Mon 10-26 CDT, fail-stop watchdog armed. Card GS (static check boots + R2 re-judge) runs right after. **Cancelled 2026-10-24 by James** (soak and Card GS); v25m was never soak-cleared.
- **Card:** [`card-v25m.png`](card-v25m.png) (base v25 card: [`card.png`](card.png)).
- **Launch:** both steps with `ROWMAP=rowmap-mixed-ar-v1.json` (the file goes in the hook directory), for example `NAME=… ROWMAP=rowmap-mixed-ar-v1.json bash launch_sidecar.sh sidecar`, then the same for `server`.
- Receipts, fitter, harness and the layout-separability test: [`results/2026-10-23-ar5-mixed-map/`](results/2026-10-23-ar5-mixed-map/).

## What this runs

DeepSeek-V4.1-Flash-1017 (MXFP4 routed experts, FP8 attention/Engram, in-checkpoint DSpark drafter) at the
full **1,048,576-token** context, on:

* **GB300** — `vllm serve` (image `sha256:3af6a45b…`, nightly-af7f9488), TP1, `--moe-backend deep_gemm_mega_moe`,
  295 of the 384 experts per MoE layer resident in HBM, Engram pinned in host memory (`{"cpu_offload": true}`),
  32 seats, `fp8_ds_mla` KV, DSpark k=5/k=3/k=3 by batch size.
* **RTX PRO 6000 Max-Q** — original-el8's peer tier-2 server (b12x kernels) holding the other **89 experts
  per layer** and computing their contribution on demand. The two processes share one `/dev/shm` buffer with
  a device-side sequence counter: the GB300 publishes a step, the card answers, the GB300 adds the result.

**As served (v27, since 2026-10-26 04:08 CDT):** the base launch plus the v25m mixed map
(`rowmap-mixed-ar-v1.json`, still 295 hot / 89 cold per layer), the 48 GiB Grace prefix-KV tier, and
`"rejection_sample_method":"block"` in the spec config. The base launch (static map, no tier, sequential
verification) is what the 2026-10-22 headline numbers were measured on.

Why it is faster: the cold-expert contribution stops being a serial C2C read and becomes a parallel GEMM on a
second GPU. Measured step time on this chassis falls from ~14.6 ms (Grace offload, C1) to ~10.6 ms, which is
where the 1.5–1.9× decode comes from. Prefill gains even more: cold rows are computed in parallel instead of
streamed, which is why the 1M first token drops from 81 s to 33 s.

**The correctness trap that shapes this whole recipe:** if the sidecar misses a deadline, the hook does not
error — it *skips that layer's cold-expert add and keeps emitting tokens*. Two mechanisms cover it: an
effectively-infinite spin limit (`VLLM_EXP_PEER_SPINS=2e9`, so a dead sidecar hangs rather than corrupts) and a
fail-closed watchdog that kills the lane on any timeout, stall or dead sidecar. The bring-up gate is a
per-layer cosine check against the GB300's own cold path.

## Hardware

One NVIDIA DGX Station GB300 with an RTX PRO 6000 Blackwell Max-Q add-in card. Profile:
[`hardware/dgx-station-gb300-rtx6000.yaml`](../../../hardware/dgx-station-gb300-rtx6000.yaml).

| | value |
|---|---|
| Compute GPU | GB300, **250.7 GiB** visible / 249.8 GiB CUDA-allocatable (not the 288 GB spec), 1,300 W cap |
| Sidecar GPU | RTX PRO 6000 Blackwell Max-Q, **95.6 GiB**, 300 W cap (325 W max), 96 GB class |
| GPU↔GPU | PCIe via the Grace root complex; **P2P unsupported** (`topo -p2p r` = NS) — hence `/dev/shm` |
| Host | 494.5 GiB LPDDR5X, Grace 72× Neoverse-V2, NVLink-C2C |
| Driver / CUDA | 595.91.07 / 13.2 (container runtime 13.0.2) |
| OS / kernel | Ubuntu 24.04.5 LTS / 7.0.0-1019-nvidia-64k |
| Power | 1,600 W Station budget shared by both GPUs, `vsloshd` inactive |

Two settings that are not optional:

1. **CDI for the RTX card.** DGX OS's `/usr/libexec/dgx-desktop-docker-gpus.sh` hides the VGA-class card from
   `--gpus`. Generate a CDI spec (`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) and pin by UUID
   (`--device nvidia.com/gpu=<RTX UUID>`). The launcher refuses to start the sidecar on a card already holding
   ≥2,048 MiB, because that card also drives the display/JetKVM.
2. **Local NVMe for the checkpoint**, and persistent JIT caches (vLLM, FlashInfer, DeepGEMM, tilelang, triton,
   b12x). The sidecar reads expert tensors out of the shards layer by layer at prep time.

## Software

| piece | pin |
|---|---|
| image (both GPUs) | `vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063` = nightly `af7f9488`, vLLM `0.30.1rc1.dev245+gaf7f9488c`; torch 2.13.0+cu130, FlashInfer 0.7.0, triton 3.7.1. The Hub tag has been rotated off — pull by digest or keep a `docker save` copy |
| hook (GB300) | [original-el8/dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research) `@87af662dd0a70f72c250c2085cf12eea2be8d882`, `deepseek-v4.1-flash/m3/hook` (Apache-2.0) — `mega_peer_hook.py`, `sitecustomize.py`, `peer_tier2.py`, `peer_fused2.py`, `ll_gemm.py`, `mxfp8_splitk.py`, `rowmap-static-v1.json` |
| sidecar server | el8 `peer_server2.py` @87af662 + [`patches/peer-server2-local.patch`](patches/peer-server2-local.patch) (path edits + ack-before-bookkeeping), run as `/sc/peer_server2_local.py` |
| sidecar kernels | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) `@2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88` (Apache-2.0) |
| overlay | vLLM [#58132](https://github.com/vllm-project/vllm/pull/58132) decoder bounded replay, 5 files bind-mounted over `dist-packages` |
| launcher lineage | [RonanLabsAI/dgx-station-frontier](https://github.com/RonanLabsAI/dgx-station-frontier) `recipes/dsv41-flash-sidecar` (Apache-2.0), adapted for Grace-only/CAND arms and our caches |

Every mounted file's sha256, read **inside the running containers**, is in
[`results/2026-10-22-v25-sidecar/raw/hook-lock.txt`](results/2026-10-22-v25-sidecar/raw/hook-lock.txt).
Checkpoint identity (revision, 49 shards, 510,311,114,235 B, config/index hashes) is in
[`raw/weights-1017-receipt.txt`](results/2026-10-22-v25-sidecar/raw/weights-1017-receipt.txt).

## Launch

Two steps, in this order — the server refuses to start unless the sidecar log says `serving`.

```bash
export GB300_UUID=$(nvidia-smi -L | awk -F'[( )]' '/GB300/{print $2}')
export RTX_UUID=$(nvidia-smi -L   | awk -F'[( )]' '/RTX PRO 6000/{print $2}')

# 1. the card: 40 layers of cold experts, 560 CUDA graphs, ~68 GB, ~46 s
bash scripts/launch-sidecar.sh sidecar

# 2. the GB300: vLLM with the hook, 295 hot / 89 cold, 1M context, 32 seats, KV pool declared at 8.5 GiB
EXTRA="--kv-cache-memory 9126805504" bash scripts/launch-sidecar.sh server
```

**The serving release, v27** — the base launch plus the v25m map, the kv-host-tier flags and block verification.
`SPEC_EXTRA` is spliced into the `--speculative-config` JSON before its closing brace. It defaults to empty, which
gives the base launch's args byte for byte.

```bash
export NAME=dsv41-vllm-1017-v27-block-BOUND-REF ROWMAP=rowmap-mixed-ar-v1.json
bash scripts/launch-sidecar.sh sidecar
EXTRA="--kv-cache-memory 9126805504 --kv-offloading-size 48 --kv-offloading-backend native" \
SPEC_EXTRA=',"rejection_sample_method":"block"' bash scripts/launch-sidecar.sh server
```

Defaults you may need: `M3_DIR` (`~/dsv41/m3` — hook checkout, overlay tree, logs, `local/peer_server2_local.py`),
`B12X_DIR` (`~/dsv41/b12x`), `MODEL_DIR`, `CACHE_DIR` (`~/dsv41/cache-a`), `ROWMAP`
(`rowmap-static-v1.json` = 295/89), `NAME`, `SEQS` (32), `GPU_UTIL` (0.97), `MAXLEN` (1048576), `PORT` (30006),
`PEER_SPINS` (2e9), `REPLAY_OVERLAY` (1), `EXTRA` (empty), `SPEC_EXTRA` (empty). `bash scripts/launch-sidecar.sh stop` stops
server-then-sidecar and keeps both containers.

**`EXTRA` carries the KV size and defaults to empty.** Without it the pool is whatever util 0.97 leaves after
vLLM's memory profile (4.9 GiB = 2,537,081 tokens) and nothing errors. With it, the boot log must say
`reserved 8.5 GiB memory for KV Cache as specified by kv_cache_memory_bytes config` and
`GPU KV cache size: 4,400,013 tokens, Maximum concurrency for 1,048,576 tokens per request: 4.20x`.
`docker start` of an existing container keeps its flags; only fresh launches need `EXTRA`.

**Bring-up gate** (do this once per new map/image — it is what proves the sidecar path is numerically right):

```bash
MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 bash scripts/launch-sidecar.sh server
python3 scripts/gate_cosine.py ~/dsv41/m3/logs/<container>.log
```

The gate reads only post-startup `MEGA_PEER check …` lines (vLLM's dummy profile run emits all-zero checks
that must be ignored) and passes on **40/40 layers, ≥32 cold rows, min cos ≥ 0.999, 0 timeouts**.

**Keep it alive** — the watchdog is part of the recipe, not an accessory:

```bash
python3 scripts/sidecar_watchdog.py --server "$NAME" --mode enforce \
  --status ~/dsv41/m3/logs/sidecar_watchdog.status.json \
  --log    ~/dsv41/m3/logs/sidecar_watchdog.log &
bash scripts/soak_check.sh <ssh-host> "$NAME"     # one health sample from another machine
```

`--rollback <container>` makes a trip auto-start the previous lane; the reference lane runs **fail-stop**
(no `--rollback`): a wrong-token risk kills the lane and leaves it down.

## Verify

| gate | how | bar |
|---|---|---|
| bring-up numerics | `scripts/gate_cosine.py` on a check-mode boot | 40/40 layers, ≥32 rows, min cos ≥ 0.999, 0 timeouts — measured **0.99906** |
| lane health | both containers up, `/v1/models`, watchdog `trip=null`, `words[2]=0`, boot KV line 4,400,013 tokens | `scripts/soak_check.sh` prints `OK` |
| KV capacity | four ~1M contexts resident at once, 16K tokens decoded each (`results/2026-10-22-kv85/scripts/kv2_coresident.py`) | 4 running / 0 waiting, 0 preemptions, 0 errors, 0 trips, 0 timeouts — measured KV peak **75.3%** |
| output quality | teacher-forced flips vs a same-map floor, BFCL dev+held-out, GSM8K-1319, 500-needle, gauntlet, reasoning-effort vocabulary, one image request | SC2's 12 pre-registered gates |
| drafter | per-class decode wall-clock, 20 cells × {C1, C8} | no cell worse than −3% vs both controls |
| speed | bracketed ladders with a control on each side | knee C1 ≥ 198, knee C16 ≥ 974, sidecar C16 ≥ Grace-only +3% |
| soak | 48 h, sampled every 5 min — restarted with the KV change: 2026-10-22 18:52 → 2026-10-24 18:52 CDT. *(Corrected 2026-10-26: that window was cut at ~25 h and the v25m window was cancelled; no 48 h soak has completed on any sidecar config, v27 included.)* | no timeout, no trip, no stall |
| v27 serving release | boot log KV line 4,400,013; `docker inspect` args contain `"rejection_sample_method":"block"`; watchdog in enforce mode with PPID 1, `trip=null` | checked at promotion (2026-10-26 04:08) and after each of the day's five restores |
| kv-host-tier (variant) | boot log `Creating v1 connector with name: OffloadingConnector` + `Created mmap file /dev/shm/vllm_offload_….mmap (51.54 GB)`, GPU KV line still 4,400,013; evict-replay probe in `results/2026-10-24-kv3-host-prefix-tier/scripts/kv3_cells.py` (needs dev mode, test boots only) | host-tier hit + load, identical output; Card KV3's 10 gates |

Nothing here is promoted on throughput alone: the cosine gate and the 0-timeout soak are pass/fail, and a
timeout is treated as a correctness bug rather than a slow request.

## Results

### v27 = the serving release (promoted 2026-10-26 04:08 CDT)

v27's measured delta over v25m-kvtier is Card BR1 (below). The promoted container got smoke, consumer and
restore checks, not a new speed bracket. The headline table at the top was measured on the v25 base launch on
2026-10-22.

### Card BR2: k ladder + peer harness (2026-10-25 21:36–23:24 CDT, `results/2026-10-26-br2-k-ladder-peerbench/`)

Bundle: [`results/2026-10-26-br2-k-ladder-peerbench/`](results/2026-10-26-br2-k-ladder-peerbench/). It holds
`throughput.csv` (per-class, mixed, tokens/step and every peerbench TG/PP cell per arm), `raw/per-position-curves.json`,
`raw/<arm>/peerbench-BR2-<arm>.md/.json`, both verdicts and the scripts, including `peerbench.py`. K9 (block + k=9)
reads 0.79–0.92× the v25m-kvtier controls at C1 and 0.88–0.93× at C8; k=5 stays. On the BRRR peer harness this lane reads 2K/C1 466
tok/s TG p50 at 5.13 tok/step and 64K prefill 46,246 tok/s. The full 12-row side-by-side is in the 2026-10-26
section above.

### Card BR1: block verification (2026-10-25 19:33–21:35 CDT, `results/2026-10-26-br1-block-verification/`)

Bundle: [`results/2026-10-26-br1-block-verification/`](results/2026-10-26-br1-block-verification/). It holds
`throughput.csv` (per-class temp-1.0 and temp-0 cells, mixed reps, tokens/step, TF flip rates), both verdicts,
`raw/<arm>/` and the scripts. At temperature 1.0, B ÷ controls is 0.996–1.054 at C1 and 1.008–1.082 at C8, with
tokens/step +2.0% and TF flips at the control floor. Per-request rows are withheld because the prompts are private
transcripts.

### KV pool declared at 8.5 GiB (2026-10-22 evening, `results/2026-10-22-kv85/`)

The lane as first promoted let `--gpu-memory-utilization 0.97` size the KV pool: 4.9 GiB after weights
(233.5 GiB), activation (3.9) and CUDA graphs (1.2). vLLM's own boot line said 10.37 GiB would "fully utilize"
the GPU. Declaring the pool with `--kv-cache-memory 9126805504` (8.5 GiB) changes nothing else — same image,
same autotune store, same kernels:

| | util 0.97 | **8.5 GiB declared** |
|---|--:|--:|
| KV tokens at 1M max-model-len | 2,537,081 | **4,400,013** (+73%) |
| 1M-token requests at once (boot line) | 2.42× | **4.20×** |
| C1 decode, real text 8.2K/1,024 | 279.5 (mean of 2 boots) | **275.1** (mean of 6, sd 5.2) |
| C16 decode | 1,576.9 | **1,575.9** |
| 1M needle, first token | 33.2 s | 35.5 s (recalled; 1M-depth decode 256 tok/s) |

The C1/C16 rows are not a same-window A/B — the kernels and tune store are identical, and the gap is inside
live FlashInfer tuning's boot-to-boot spread.

**Stress, on the live lane:** a ~1M needle + a cold 131K prompt + C16 all at once (all pass), then four ~990K
prompts each decoding 16,000 tokens so all four are resident together — **4 running / 0 waiting, KV usage peak
75.3%**, 0 preemptions, 0 engine errors, 0 watchdog trips, sidecar timeout counter unchanged at 0. The GB300 sat
at **255,308 of 256,703 MiB** at the peak (1,395 MiB free). That margin is why we stopped at 8.5: 9.5 GiB would buy
+0.5M tokens and leave ~370 MiB, and an out-of-memory on this lane is a crash, not a slowdown.

The speed cells use **RonanLabsAI's `kbench_chat_ds.py`**, the same harness as every other real-text number in
this recipe.

Side finding: a cold ~1M prefill holds up the queue — the 131K prompt next to it waited 34 s for its first token
and the C16 aggregate fell to 718 tok/s while the 1M prompt was read. See Known limits.

### The sidecar round (`results/2026-10-22-v25-sidecar/`)

Bundle: [`results/2026-10-22-v25-sidecar/`](results/2026-10-22-v25-sidecar/) — `throughput.csv` (every cell
with its arm and receipt), `power-by-phase.csv`, `system.json`, and the raw verdicts.

**Decode, real text (8.2K in / 1,024 forced out, thinking off, 0% prefix hits)** — bracketed
CTL1 → CAND1 → CAND2 → CTL2 on one box, sidecar values are the mean of the two independent boots:

| cell | incumbent v23 | sidecar | Δ |
|---|--:|--:|--:|
| C1 | 187.5 | **279.5** | +49% |
| C4 | 321.7 | **693.2** | +116% |
| C16 | 524.5 | **1,576.9** | 3.0× |
| C32 | 936.9 | **2,400.3** | 2.6× |
| TTFT 8.2K | 0.66 s | **0.19 s** | 3.5× lower |
| acceptance (C1) | 2.955 | 2.961 | +0.5% |

**Agent-text knee** (C1 222.8 vs 192.0, +16%; C8 891.1 vs 598.6/601.9, +49%; C16 1,414.5 vs 937.0, +51%) and
**per-class C1** (prose 294.8 vs 232.3, structured 299.1 vs 250.3, code 408.9 vs 293.4, shell 413.7 vs 316.9,
tool-JSON 367.2 vs 300.0) are in `throughput.csv`. Note the per-class fixture reads higher than the 8.2K
harness on every arm — do not mix the two.

**Long context (SC5):** 999,303-token prompt → first token in **33.23 s** (30,069 tok/s), needle recalled;
128K cold prefill 2.61 s; C16 after 1M 32/32 at 1,565.9 agg; C1 decode after 1M 276.5 vs 279.5 before. KV pool
2,537,081 tokens = 2.42× one 1M request at util 0.97 (superseded: 4,400,013 / 4.20× with the declared 8.5 GiB
pool above).

**Against Ronan's published lane — context, not a competition.** On the cells his 1017 sidecar log publishes for
the same harness, this lane reads C1 279.5 vs 273.6, C4 693.2 vs 689.1, C16 1,576.9 vs 1,545.5, C32 2,400.3 vs
2,276.6 (+2% / +1% / +2% / +5%). Same mechanism, same chassis; the differences are our 295/89 map (his 285/99),
the ack-order fix and live-tuned kernels. That is why [Credits](#credits) leads with him.

**Quality (SC2, all 12 gates PASS):** teacher-forced flips inside the same-map control floor; BFCL dev
92.67 vs 92.33 and held-out 77.65 vs 77.88; GSM8K-1319 96.97 = incumbent; needle ladder 500/500 exact;
gauntlet 12/12; reasoning-effort vocabulary and an image request both pass; 20/20 per-class decode cells
within −3%; 20-minute soak with **0 sidecar timeouts**.

**Placement (SC3):** the nested 285/99 map would buy ~6.09M KV tokens but **failed its pre-registered cosine
bar** (0.99885 vs 0.999 on two layer-39 T=128 checks), so the uniform 295/89 map serves. Its speed with the
sidecar is unmeasured.

**Power (SC5):** GB300 module p99 **1,353 W** during 64K prefill, 1,263 W at C32 decode; `sw_power_cap` active in
89.5% (64K prefill) / 85.7% (C32 8K prefill) / 2.4% (C32 decode) of 1 s samples; SM clock p50 drops to 1,980 MHz
in prefill (2,062 incumbent); RTX p99 ≤ 194 W against its 300 W cap. Full table in `power-by-phase.csv`.

## Known limits

* **Block verification (Card BR1) only acts at temperature > 0.** vLLM af7f9488 skips it on greedy rows
  (`USE_BLOCK_VERIFICATION and not is_greedy`), so a temperature-0 bench cannot measure it. It reads as a no-op
  there, and that is what Card BR1's temp-0 window showed. The −0.4…+5.4% C1 (per class) and +2.0% tokens/step were measured at
  temperature 1.0 / top_p 1.0 with 40 private transcript prompts per class. Clients that send temperature 0 get
  nothing from it. **Promoted as v27 on 2026-10-26** (one key on the v25m-kvtier launch; rollback = the
  `-RETIRED-REF` pair).
* **Block verification is lane-specific:** it needs probabilistic drafting, and it was not clean on the
  Grace-only sibling (BR3 null at 0.999× tokens/step with a greedy drafter; with probabilistic drafting BR4 PB,
  BR5 and BR6 each tripped a temperature-0 parity gate, and it was retired there). Do not copy this key to
  another lane without its own bracket.
* **v27 serves on the v25m map, which missed the bring-up cosine gate** (Card GV: min cos 0.99811 vs the 0.999
  bar, 1 of 1,439 gated checks, at layer 39). It passed the other 9 gates, and it was put on `:30006` as
  v25m-kvtier and then v27 with that miss open. The pre-registered R2 re-judge
  (`results/2026-10-26-gs-cosine-r2/`) has not run. The 0.99906 pass in Verify is the v25 static map.
* **k > 5 is a loss on real traffic with this drafter (Card BR2).** k=9 (with block verification) reads 0.81–0.92×
  at C1 per class and 0.79× on mixed traffic. The per-position curves show why: marginal acceptance drops from
  0.61–0.82 (positions 1–5) to 0.51–0.67 from position 7, and position-1 acceptance itself falls (prose .72 → .61).
  The checkpoint's `dspark_block_size` is 5. k=9 only paid on padded single-stream codegen (peer harness 64K/C1).
  BR2 did not isolate k from block verification, because its controls ran sequential verification.
* **A k change does not force an autotune retune on this image.** `num_speculative_tokens` is not in the autotune
  hash on af7f9488, and K9 booted as a hash hit in 529 s. The card's brief had expected a 25–35 min retune. The
  README and limits carried no retune claim to strike.
* **Peer-harness numbers are 2 rounds** (the peer's published table is 5), with no cool-down wait, on our sampling
  path. Their column is copied from their model card, not re-run. A2 agrees with A within 2.5% on TG. PP/TTFT at
  2K/C1 moved 8–9% between A and A2.
* **kv-host-tier variant (Card KV3):** the card's launcher ran with `VLLM_SERVER_DEV_MODE=1` (smoke only, for
  `/reset_prefix_cache`), and production drops it, so that exact boot was not measured on its own. It was measured on the v25m map. ~31M
  prefix tokens = 48 GiB / 1,656 B per hit token on this lane only. Speed-neutral on the gated C1/TTFT/warm
  cells, but real multi-user hit rates, eviction pressure and C>1 with the tier were not measured. Fidelity on the tier arm is
  identical short probe outputs + gauntlet 12/12 + needle; there was no TF/GSM8K/BFCL run on it. Host MemAvailable drops by ~48 GiB
  (min 129.8 GiB). vllm#47890 did not reproduce, but only two prompt lengths were probed. LMCache would need the hybrid
  KV manager off.
* **No 48 h soak has completed on any sidecar config.** The 2026-10-22 window was cut at ~25 h, and James cancelled
  the v25m window on 2026-10-24. The recipe stays experimental.
* **The power cap binds in prefill** — and it bound on the incumbent too (65% / 52% of samples). The SC5 bars
  (<1% throttled, module ≤1,190 W) were set too tight for this chassis; every number here was measured under
  that throttling. A 240 V / uncapped retest is the obvious next measurement.
* **A silent sidecar timeout is wrong output.** Infinite-spin + watchdog, or do not serve. Do not run the lane
  in `--mode report` and call it production.
* **Uniform rowmap only**, unless you port el8's per-layer-plan change to `peer_server2`. Non-uniform maps also
  change the cold path's cost; ours was never speed-measured because its gate failed first.
* **Effort mapping moved:** this image maps `low`→50, `high`→75 (default), `max`→100; the GB300-only lane's
  0909 image maps `low`→25, `high`→50. Clients that send `low` think twice as long after a move to this lane.
* **One runaway is possible at temp 0** (GSM8K item 1304 hit the 131,072 cap once; a 32K cap probe read 0/32
  and pinned efforts 0/8). Attributed to the image's effort-75 tail, not the sidecar; a client `max_tokens` is
  the only real guard.
* ~~**Per-class acceptance is not measured** in this round (blended 2.96 tokens/step only).~~ **CORRECTED
  2026-10-26:** measured at temperature 1.0 in Card BR2 (C1 tokens/step: prose 2.85, structured 3.03, code 3.83,
  shell 3.98, tool-JSON 3.63; window blend 3.18–3.22), with per-position curves.
* **Live FlashInfer autotune**, no pinned tune file: a few percent of C1 varies boot to boot, and the autotuner
  reports some `mxfp8_gemm` shapes falling back to `tactic=-1`.
* **Replay ON vs OFF is one rep**: agent replay r2 612.1 OFF vs 480.3 ON, but TTFT is better ON (0.19 vs 0.29 s).
  The lane runs ON. The +27% is not a law until repeated.
* **vLLM v0.31.0 was NOT adopted** — quality identical, but knee C8 read −3.8% and the runaway cell tripped.
* **KV is declared, so vLLM no longer checks activation headroom at boot.** Measured margin: 1,395 MiB free on
  the GB300 at the stressed peak. Vision requests and C32 prefill bursts were not in that stress. 9.5 GiB was
  declined (~370 MiB would remain).
* **`EXTRA` defaults to empty** — a fresh launch without `EXTRA="--kv-cache-memory 9126805504"` silently serves
  2,537,081 tokens. Check the boot's KV line.
* **A cold 1M prefill stalls everything behind it** (131K neighbour: 34 s to first token; C16 aggregate 1,576 →
  718 tok/s while it ran). Keep long batch jobs off the interactive lane, or accept the stall.
* **KV routes not taken:** KV quantization (`nvfp4_ds_mla`) changes numerics; more cold experts on the card
  (89 → 99/109 per layer, +3.6M/+7.3M tokens) is modeled at −5 to −11% C1; HiSparse host-resident KV exists in
  this image but is not wired for DeepSeek-V4.1 or the MegaAttn backend.
* **Boot 527 s warm**; 46–50 s of it is the sidecar. Cold JIT caches are ~30 min.
* **The Docker Hub tag is gone**; the digest is the only reliable pull.

## Rollback

```bash
# 1. stop the guard first (it would otherwise see a dead server and trip)
pkill -f sidecar_watchdog.py
# 2. server, then sidecar
bash scripts/launch-sidecar.sh stop
# 3. bring the previous lane back (kept, stopped, same box): v25m-kvtier = v27 without the block key
docker start dsv41-vllm-1017-v25m-kvtier-RETIRED-REF-rtx   # sidecar first; wait for 'serving' in its log
docker start dsv41-vllm-1017-v25m-kvtier-RETIRED-REF
```

*(Corrected 2026-10-26: step 3 named `dsv41-vllm-1017-v23-pinexact-RETIRED-REF`, the Grace-offload v23 lane. Since
the v27 promotion the previous lane is the v25m-kvtier pair above. v23 stays kept as the older fallback:
`docker start dsv41-vllm-1017-v23-pinexact-RETIRED-REF` binds in ~260 s.)* On a fresh box, undo only block
verification by relaunching the server step with `SPEC_EXTRA` empty.

To undo only the KV change, start the kept util-0.97 pair instead (sidecar first, then server):
`docker start dsv41-vllm-1017-v25-sidecar-kv4.9-KEPT-rtx && docker start dsv41-vllm-1017-v25-sidecar-kv4.9-KEPT`
(it bound in ~7 min on the reference box).

To drop only the kv-host-tier variant, relaunch the server step without the two tier flags
(`EXTRA="--kv-cache-memory 9126805504"`). `docker start` of a tier container keeps its flags. The mmap is unlinked
at boot, so stopping the container frees the 51.54 GB.

With no previous lane on the box, the fallback is the GB300-only sibling recipe
([`../deepseek-v4.1-flash-vllm-uva-dspark`](../deepseek-v4.1-flash-vllm-uva-dspark)): same checkpoint, Grace
offload, no RTX card. The watchdog's fail-stop policy means a trip leaves this lane down on purpose — bring the
incumbent back by hand after reading `sidecar_watchdog.log`.

## Credits

This recipe is a productionised descendant of someone else's recipe, and the ordering here is deliberate.

* **[RonanLabsAI](https://github.com/RonanLabsAI/dgx-station-frontier) — the sidecar recipe this lane runs.**
  The two-step `sidecar` → `server` deployment shape, the launcher we adapted line by line, the b12x peer
  tier-2 protocol in production form, the `rowmap-mix-v1` calibration and its Stage-0 variants, the
  `MEGA_PEER_CHECK` agreement gate as a bring-up requirement, the 1M long-context work, and the
  `kbench_chat_ds.py` harness every real-text number here was measured on are his (`recipes/dsv41-flash-sidecar`
  plus `results/sidecar-ab.md`, `results/long-context-1m.md`, `logs/dsv41-1017/*`, Apache-2.0). His published
  1017 sidecar cells on that harness — C1 273.6 / C4 689.1 / C16 1,545.5 / C32 2,276.6 — are the bar we
  measured our own lane against; we land within a couple of percent of them (see Results), which is the
  honest read: **the mechanism is his.** The KV round (`results/2026-10-22-kv85/`) was measured on his harness
  and his CNN/DM prompt pool too.
* **[original-el8](https://github.com/original-el8/dgx-station-gb300-research)** — the M3 hook (hot/cold split
  inside DeepGEMM MegaMoE, cold path on the FlashInfer TRT-LLM kernel), `peer_tier2`/`peer_fused2`, and the
  `rowmap-static-v1.json` map this recipe ships unmodified, @87af662, Apache-2.0. Without the hook there is no
  hot/cold split to give the second GPU.
* **[b12x](https://github.com/local-inference-lab/b12x)** (Apache-2.0) — the FP4 MoE kernels the sidecar
  replays out of its 560 captured graphs.
* **vLLM** — [#58132](https://github.com/vllm-project/vllm/pull/58132) bounded decoder replay, the
  `deep_gemm_mega_moe` backend, and the UVA offloader the sibling recipe uses.
* **[Fraser Price](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR)** — the BRRR recipe for
  DeepSeek-V4.1-Flash-1017 on 4× RTX PRO 6000 (fused decode kernels, `decoder.cu`), its published results table, and
  `bench.py`, which our `peerbench.py` ports with his credit header (Apache-2.0). His table is the "theirs" column in
  the 2026-10-26 side-by-side.
* **Sun, Mendlovic, Leviathan, Aharoni, Ro, Beirami, Suresh** —
  [Block Verification Accelerates Speculative Decoding](https://arxiv.org/abs/2403.10444) (ICLR 2025), the algorithm
  behind vLLM's `rejection_sample_method: block`, which Card BR1 measured.

**Our delta, precisely** (so nobody re-credits us with the idea): the uniform 295/89 map choice and the
measurement that 285/99 fails its pre-registered cosine bar; running the agreement gate on real prompts and
publishing the check distribution; the ack-before-bookkeeping patch to `peer_server2.py`
([`patches/peer-server2-local.patch`](patches/peer-server2-local.patch)); the fail-closed watchdog and the
fail-stop serving policy; the bracketed control→candidate→control discipline on every speed cell; and the
quality gates (teacher-forced flips, BFCL, per-class decode) applied to a sidecar lane at all. Measurements,
gate design and the write-up: **J&M Recipes** (James Meadlock, Milo). Session note: the numbers were produced
with a Claude Opus 5.5 orchestrator on the research side; the recipe was written from the receipts.
