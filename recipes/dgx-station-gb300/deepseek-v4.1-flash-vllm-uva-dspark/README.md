# DeepSeek-V4.1-Flash at 1M context on one DGX Station GB300

> **Status, 2026-10-26: this recipe's release is now `v28-prob`, and `:30006` still does NOT serve it.**
> v28-prob = v26 "Seeded Adaptive" + one speculative-config key, `"draft_sample_method":"probabilistic"` (Card BR4,
> PASS). Its container `dsv41-vllm-1017-v28-prob-BOUND-REF` was booted once on 2026-10-26 (10:49–10:57 CDT: bound after 476 s, KV 2,481,694 tokens, smoke and tool call OK), verified
> and stopped-and-kept; `dsv41-vllm-1017-v26seed-BOUND-REF` is kept as its rollback. **This is a reference update,
> not a routing change.** `:30006` has served the **sidecar sibling's** pair since 2026-10-24 23:16 CDT: v25m-kvtier
> until 2026-10-26 03:58 CDT, and **v27 (+ block verification)** since 04:08 CDT
> ([`../deepseek-v4.1-flash-vllm-uva-dspark-sidecar`](../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/README.md)). v26 served this recipe on `:30006` only from 2026-10-24 ~21:30 to 23:16 CDT.
> Latest round: **Card BR5, block verification re-bracketed against v28-prob = FAIL-G6 (instrument): every speed and
> fidelity gate passed (+2.5% tokens per verify step), replay_turns tripped the opposite way from BR4; PB retired
> pending BR6.** The reference stays v28-prob.
> Before it: **Card BR4, probabilistic drafting = PASS, promoted as the reference v28-prob; + block verification
> = candidate** (retired by BR5). Before that, Card BR3 (block verification with greedy drafting) was a null.

**2026-10-24 ~21:30 CDT — v26 "Seeded Adaptive" promoted** (on `:30006` until 23:16 CDT that night; see the status above).
v26 is the v23 launch (same image, same flags), plus three changes:
- **adaptive expert residency** — hook v26ar.2;
- **a seed boot map** — the top 295 experts per layer, taken from the live counter's own dump;
- **the shipped swap policy**, with adaptation unfrozen.

With maps frozen on held-out prompts, mixed C1 is **183.7 vs 127.1 tok/s (×1.445)**. The adaptive steady state is
**218.9 tok/s agent / 189.6 JP/DE** at C1. Gauntlet is 12/12 and greedy parity 49/50. **The 48 h soak was waived by
James at promotion** — see Known limits. Receipts: [`results/2026-10-24-v26-seeded-adaptive/`](results/2026-10-24-v26-seeded-adaptive/README.md).

From 2026-10-22 to 2026-10-24 the lane ran the **sidecar sibling**
([`../deepseek-v4.1-flash-vllm-uva-dspark-sidecar`](../deepseek-v4.1-flash-vllm-uva-dspark-sidecar)). It uses the
same checkpoint and image, plus the chassis' RTX PRO 6000 computing the 89 cold experts per layer: C1 279.5 vs
187.5 tok/s for v23 here, C16 3.0×. It is still the faster lane on a box that has that second GPU; its 2026-10-24
round adds a 48 GiB Grace host prefix-KV tier. This recipe is the reference for a GB300 with no second GPU.

**Before v26 — Pin Exact, v23** (serving 2026-10-18 → 2026-10-22; v26 launches from it) (v22 + exact-size pinned host allocations: pinned host 328.0 → 252.8 GiB, boot-guard margin 12 → 198 GiB, decode and greedy non-tool output unchanged; **188.2 tok/s C1 real text, accept 2.99/step**; BFCL held-out 77.65; see the 2026-10-18 round) · rollback **v22** (k=5 band to 8 sequences, 2026-10-17) · **candidate, not promoted:** vLLM nightly-39983a5e "config C" passed all seven pre-registered gates (+6% C1, TTFT 8K 0.43 s, KV +55%; GSM8K exactly at its bar), pending James · previous **Fused Split, v21** (2026-09-27: v18 launch on the 0909 image + hook v21 — `split_ids()` as one Triton kernel; **182.8 tok/s C1 prose, +6.2 % vs v18 bracketing controls 171.8 / 172.5**, C8 687 (+4.6 %), C16 993 (+4.0 %), non-tool greedy 16/16 vs both controls; held-out BFCL on v21 pending; see Round 12) · documented release **Clean Nightly, v20** (2026-09-21: vLLM nightly `2671fedf` — the last before the #56633 mHC-fold drift — + v15 hook + off54 + fp8_ds_mla + v18's 24-seat/cudagraph flags; **181 tok/s C1 prose (+5.1% pair mean vs v18)**, KV 2.55M tokens, C24 warm agent p95 2.4–2.8 s vs 3.4; **held-out BFCL live sibling 78.8% vs v18 78.9%**, dev 92.7–93.5%; tools 64/64 ×3; $0.008 per 1000 solved tasks; see Round 11b) · rollback **Many Seat, v18** (2026-09-18: v15 hook + `--max-num-seqs 24` + token-sized `--cudagraph-capture-sizes`; 172 tok/s C1 prose, KV 2.50M tokens, C24 warm agent turn 0.55 s p50; **BFCL v4 tool-call exact-match 93.3%**) · **v19 "Nightly" was promoted and reverted on 2026-09-20** (see Round 10) · previous **v15 Pin Hot Experts** (retired 2026-09-18) · **v14 Sixty-K Agent** (retired 2026-09-17) · Historical **v13: 90 tok/s single-stream prose · 140–160 tok/s on agent/code text · 429 agg tok/s at C16** (v12: 89 / 140–160 / 311 · v11: 82 / 130–150 / 287) · 972K-token prompt prefilled in 85 s · Hermes tool-calling 10/10


## 2026-10-26 round — Card BR5: block verification re-bracketed against v28-prob = FAIL-G6 (an instrument finding); PB retired pending BR6

**What changed.** BR4's candidate PB, bracketed alone against the new reference as BR4 asked:
- **A / A2** = v28-prob (v26 + `"draft_sample_method":"probabilistic"`), byte-identical to
  `dsv41-vllm-1017-v28-prob-BOUND-REF` (G0);
- **B** = v28-prob + `"rejection_sample_method":"block"`.

Same cells, gates and bars as BR3/BR4, **pre-registered before launch**
([`PREREG-BR5.md`](results/2026-10-26-br5-block-rebracket/PREREG-BR5.md)): a trip in any class is a FAIL;
"REPRODUCED" = the same cell trips in the same direction; if G6 fails again, PB is retired on this lane until a third
design with more replay passes exists. Bracket A → B → A2, 2026-10-26 11:33–13:52 CDT:
- every arm was a hash-hit boot, ready at 474–475 s; KV 2,481,694 tokens on all three arms;
- adaptive residency was unfrozen (GO + `ctrl freeze=false`) on every arm;
- 0 foreign requests reached any arm; the production v27 pair was restored with its watchdog (PPID 1) at 13:51.

Receipts: [`results/2026-10-26-br5-block-rebracket/`](results/2026-10-26-br5-block-rebracket/README.md).

**Verdict: `FAIL`, suffix `BR5FAIL-G6`** (pre-registered, not re-scored). Nine of ten gates passed, including the
G2b temperature-0 parity gate that PB failed in BR4. G6 replay_turns failed.

| temp 1.0, B ÷ mean(A, A2) | B | ctrl |
|---|--:|--:|
| accepted tokens per verify step (per-class window) | **1.025** (3.513) | 3.427 (A 3.417, A2 3.437) |
| mixed traffic tok/s, C1 (`ar_mixed_t.py`, 3 × 120 s) | 1.009 (182.54) | 180.96 |

| temp 1.0, decode tok/s B ÷ ctrl | C1 | C8 |
|---|--:|--:|
| prose | 1.013 | 1.041 |
| structured | 1.025 | 1.030 |
| code | **1.054** (286.8 vs 272.3 tok/s) | 1.050 |
| shell_ops | 1.039 | 0.985 (bar −2.0%) |
| tool_json | 1.017 | 1.008 |

Per-position acceptance: B .787/.602/.462/.365/.297 vs control mean .782/.585/.441/.344/.275.

**Parity re-check (BR4's two trips).**
- **G2b** temperature-0 shell_ops C1: B +2.4% against a 10.3% bar (control r1/r2 spread 6.9%), and inside BR4's
  fixed 2.6% bar as well. **The BR4 trip (+3.6% vs 2.6%) did not reproduce.** All five classes passed.
- **G6** replay_turns: B per pass [6, 7]/12 against a control range of 8–9 (A [9, 8], A2 [8, 8]). **FAIL, in the
  opposite direction from BR4** (PB [8, 9] against 6–8).

**Fidelity is unchanged.** Teacher-forced top-1 flips B-vs-A 1.072% and B-vs-A2 1.064% against an A2-vs-A floor of
1.075%; margin-weighted flips 0; gauntlet 12/12 on all arms; 0 errors, 0 foreign requests.

**G6 history** (tool turns emitted per pass, of 12; temperature 0, `replay_turns.py`):

| card | arm | config | pass 1 | pass 2 | gate bar (range of the 4 control passes) |
|---|---|---|--:|--:|---|
| BR4 | A | v26 (greedy drafting) | 6 | 7 | control |
| BR4 | P | + probabilistic | 6 | 8 | 6–8 → PASS |
| BR4 | PB | + probabilistic + block | **8** | **9** | 6–8 → FAIL (more) |
| BR4 | A2 | v26 (greedy drafting) | 8 | 8 | control |
| BR5 | A | v28-prob | 9 | 8 | control |
| BR5 | B | v28-prob + block | **6** | **7** | 8–9 → FAIL (fewer) |
| BR5 | A2 | v28-prob | 8 | 8 | control |

**Instrument diagnosis.** Across the five probabilistic-drafting arms the ten per-pass counts are
6, 6, 7, 8, 8, 8, 8, 8, 9, 9 (all seven arms: 6 ×3, 7 ×2, 8 ×7, 9 ×2). The two block-on arms sit at both ends; the
block-off arms span 6–9 by themselves. The count does not follow the block key, which at temperature 0 never touches
a greedy row. A range from two controls × two passes is too narrow a bar for a temperature-0 replay that is not
deterministic on this lane: the controls alone spanned 6–8 in BR4 and 8–9 in BR5. This reading fits every receipt;
it is not a tested result, and the pre-registered verdict stands.

**Decision.** Per the pre-registration, **PB is RETIRED on this lane** pending **Card BR6** (queued 2026-10-26): the
same bracket with a redesigned G6 — 4 replay passes per arm, a pooled-control bar ±1, a control-agreement validity
check, and a per-transcript annotation. **The reference stays v28-prob** (unchanged); `:30006` keeps serving the
sidecar sibling's v27 pair.

The speed and fidelity story for block-on-probabilistic is consistent across three cards: +2.0% tokens per verify
step on the sidecar lane (Card BR1), +1.5% on top of P (BR4), +2.5% here, with TF flips at the control floor each
time:
[BR4 bundle](results/2026-10-26-br4-probabilistic-drafting/README.md),
[BR3 bundle](results/2026-10-26-br3-block-verification-null/README.md),
[BR1 bundle](../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/README.md),
[blog](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html).

## 2026-10-26 round — Card BR4: probabilistic drafting = PASS, promoted as the reference v28-prob; +block = candidate

**What changed.** Two candidates against the v26 lane, each one key in `--speculative-config`:
- **P** = v26 + `"draft_sample_method":"probabilistic"`: the DSpark drafter samples its proposals instead of taking
  the argmax;
- **PB** = P + `"rejection_sample_method":"block"` (block verification, Sun et al.).

Both keys came to this lab from the config of
[fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR).
Everything else is the v26 lane as booted: image 0909, `launch_br3.sh` (= `launch_ar6.sh` + `SPEC_EXTRA`), k-schedule
`[[1,8,5],[9,24,1]]`, 24 seqs. The controls draft greedy, as v26 does. Bracket A → P → PB → A2, 2026-10-26
07:23–10:28 CDT:
- every arm was a hash-hit boot, 474–485 s;
- adaptive residency was unfrozen (GO + `ctrl freeze=false`) on every arm;
- 0 foreign requests reached any arm; the production v27 pair was restored with its watchdog at 10:28.

Receipts: [`results/2026-10-26-br4-probabilistic-drafting/`](results/2026-10-26-br4-probabilistic-drafting/README.md).

**Verdict: `PASS-P`, suffix `BR4-PPASS_PBFAIL-G2b-G6`** (pre-registered, not re-scored). P passed all ten gates. PB
passed every speed and fidelity gate and failed two temperature-0 parity gates (explained below).

| temp 1.0, ÷ mean(A, A2) | **P** | PB | PB ÷ P |
|---|--:|--:|--:|
| accepted tokens per verify step (per-class window) | **1.103** (3.438 vs 3.119) | 1.120 (3.491) | 1.015 |
| mixed traffic tok/s, C1 (`ar_mixed_t.py`, 3 × 120 s) | 1.007 (181.09 vs 179.86) | 1.014 (182.45) | 1.008 |

| temp 1.0, decode tok/s ÷ ctrl | C1 **P** | C1 PB | C8 **P** | C8 PB |
|---|--:|--:|--:|--:|
| prose | **1.101** | 1.114 | **1.091** | 1.126 |
| structured | **1.109** | 1.139 | **1.082** | 1.122 |
| code | **1.045** | 1.078 | **1.099** | 1.130 |
| shell_ops | **1.113** | 1.126 | **1.132** | 1.157 |
| tool_json | **1.162** (294.0 vs 253.0 tok/s) | 1.154 | **1.187** (791.8 vs 666.9) | 1.170 |

Per-position acceptance (temp-1.0 window, accepted at draft position ÷ drafts):

| position | 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|
| control, mean(A, A2) | .699 | .501 | .378 | .299 | .245 |
| **P** | **.785** | **.587** | **.443** | **.346** | **.278** |
| PB | .786 | .598 | .457 | .360 | .290 |

**Fidelity is unchanged.**
- Teacher-forced top-1 flips: P 1.10–1.13%, PB 1.10%, against an A-vs-A2 control floor of 1.136%; margin-weighted
  flips 0 on every compare.
- Gauntlet 12/12 on all arms; P's replay_turns tool-turn emission [6, 8]/12 is inside the control range 6–8.
- P's temperature-0 window is a no-op on every class (G2b PASS).
- 0 errors, 0 foreign requests.
- **KV 2,481,694 tokens on P and PB vs 2,495,694 (−0.56%).** With probabilistic drafting vLLM allocates a bf16
  draft-logits cache (24 × 5 × 129,280 × 2 B ≈ 31 MB) before KV profiling. The brief predicted the drop before the run
  and the G0 rule allowed up to 1.5%.

**Why PB is a candidate and not the release.** Both of its failures are on temperature-0 PARITY gates, where neither
key touches a greedy row:
- **G2b** shell_ops C1 read +3.6% against a 2.6% bar (1.5 × a 1.7% control spread);
- **G6** replay_turns emitted tool calls on 9/12 tool turns in one pass, against a control range of 6–8.

Both trips are in the "faster / more" direction. We read them as run-to-run nondeterminism against two tight bars,
but the pre-registered verdict stands and is not re-scored. PB will be re-bracketed alone against v28-prob later.

**Mechanism (our reading).** With a temperature > 0 target, the drafter's samples match the target's distribution
far better than its argmax does: acceptance rose by 0.03–0.09 at every draft position. Greedy drafting was this
lane's missing lever. It also explains Card BR3: block verification only pays once drafting is probabilistic. It
gained nothing here with greedy drafts (BR3, 0.999× tokens per step), +1.5% on top of probabilistic drafts (PB ÷ P),
and +2.0% on the sidecar lane, which already drafts probabilistically (Card BR1):
[BR3 bundle](results/2026-10-26-br3-block-verification-null/README.md),
[BR1 bundle](../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/README.md),
[blog](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html).

**Decision:** **P is promoted as this recipe's reference, `v28-prob`**
(`dsv41-vllm-1017-v28-prob-BOUND-REF`, booted once by `park_v28_prob.sh` with the production soak `/ar` bind,
verified, stopped-and-kept; park facts OK — KV 2,481,694, probabilistic key in Args, smoke and tool call OK; the lane was dark 10:49–11:06 CDT and the v27 pair plus watchdog were restored). v26seed is kept as the rollback. `:30006` keeps serving the
sidecar sibling's v27 pair. **PB stays a candidate.**

## 2026-10-26 round — block verification on v26 (Card BR3): a NULL, NOT promoted

**What changed.** One key was added to the v26 speculative config: `"rejection_sample_method":"block"` (block
verification, Sun et al.). The idea came to this lab via
[fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR).
Everything else is the v26 lane as booted: image 0909, `launch_ar6.sh` + `SPEC_EXTRA`, **greedy drafting**,
k-schedule `[[1,8,5],[9,24,1]]`, 24 seqs. The run was bracketed A → B → A2, 2026-10-26 04:26–06:49 CDT:
- every arm was a hash-hit boot, 474–484 s;
- KV was 2,495,694 tokens on all arms;
- adaptive residency was unfrozen (GO + `ctrl freeze=false`) on every arm.

Receipts: [`results/2026-10-26-br3-block-verification-null/`](results/2026-10-26-br3-block-verification-null/README.md).

**Verdict: FAIL `BR3FAIL-G2`** (pre-registered). Structured decode at temperature 1.0 read C1 0.973 and C8 0.970,
against a −2% bar (−2.1% at C1). **The headline is a null, not a regression.**

| temp 1.0, B ÷ mean(A, A2) | C1 | C8 |
|---|--:|--:|
| accepted tokens per verify step (per-class window) | **0.999** (3.116 vs 3.118) | — |
| mixed traffic tok/s (`ar_mixed_t.py`, 3 × 120 s) | 0.998 (−0.2%) | — |
| prose | 1.000 | 0.989 |
| structured | **0.973** ✗ | **0.970** ✗ |
| code | 0.984 | 1.088 |
| shell_ops | 0.995 | 1.044 |
| tool_json | 1.015 | 1.086 |

The C8 code and tool_json rises are within the controls' own C8 spread: A and A2 differ by 4.2% on code and 9.1% on
tool_json at C8.

**Fidelity is unchanged.**
- Teacher-forced top-1 flips are 1.09–1.13%, against an A-vs-A2 control floor of 1.09%; margin-weighted flips 0.
- Gauntlet 12/12 on all arms.
- The temperature-0 window is a no-op on every class (G2b PASS).
- 0 errors, 0 foreign requests.

**Mechanism (our reading, not tested by a receipt).** This lane drafts greedy: there is no `draft_sample_method` in
its config. Block verification reweights the draft distribution, and a one-hot draft leaves it nothing to reweight,
so it degenerates to standard verification.

The contrast is the sidecar lane. It already drafts probabilistically, and the same key gained +2.0% tokens/step
there (Card BR1, promoted as v27):
[BR1 bundle](../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/README.md),
[blog](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html).

**Decision:** not promoted on this lane; v26 stays as published. **Next: Card BR4.** It tests probabilistic drafting
alone, then probabilistic + block, in the same bracket (queued 2026-10-26). *(It ran the same morning: probabilistic
drafting PASS, promoted as the reference v28-prob; see the round above.)*

## 2026-10-24 round — v26 "Seeded Adaptive": seed map + adaptive residency, PROMOTED (48 h soak waived)

**Config.** The v23 launch is unchanged: 0909 image, the same serve flags, and autotune hash `62426808` (`Loaded 189`
on both boots). The changes are:
- **Hook v26ar.2.** This is the published v26ar.1 hook plus env-gated telemetry only: a per-step ring
  `PIN_STEP_RING=65536`, swap-apply timestamps, and CUDA events `PIN_SWAP_CUDA_EVENTS=1`.
- **Boot map `PIN_ROWMAP=/w/rowmap-mixed-ar-v1.json`.** This is the top 295 experts per layer from one cumulative
  dump of the live counter (2026-10-23). The 295 hot / 89 cold geometry is unchanged.
- **The shipped policy, adaptation unfrozen:** drain every 64 steps, ≤ 2 swaps per layer, half-life 500 steps,
  hysteresis 1.5×, min 8, ≤ 8 swaps per barrier.

Promoted by James on 2026-10-24 at ~21:30 CDT, **with the 48 h soak explicitly waived**. Write-up:
[Adaptive Residency, Part 3: The Seed Was the Prize](https://al-engr.com/gb300-adaptive-residency-seed-map.html).
Receipts: [`results/2026-10-24-v26-seeded-adaptive/`](results/2026-10-24-v26-seeded-adaptive/README.md). Card AR6 v2 verdict: `AR6-H1PASS-H2FAIL`.

**Design.** Two boots in one window: September static map vs seed map. Each boot ran a frozen-map held-out mixed run,
then agent → JP/DE → agent shift passes. The shipped policy (B0) ran against 2× (B1, drain 32) and 4× (B2, drain 16)
queue rates, interleaved over 3 rounds.

**H1 — the seed map (PASS).** Adaptation frozen on both arms, held-out prompts, same window:

| C1 | Sept static map | seed map | Δ |
|---|---:|---:|---:|
| mixed aggregate (post-60 s) | 127.12 | **183.73** | **×1.445** (bar ×1.2) |
| JP/DE per-request median | 85.45 | 177.23 | ×2.07 |
| agent English per-request median | 202.74 | 201.36 | −0.7 % (bar ±2 %) |
| first JP/DE request after boot | 84.96 | 177.89 | ×2.1 |

With adaptation on, both boots converge to the same steady state: **JP/DE 189.6 / agent 218.9 tok/s** C1 on the seed
boot, against 189.9 / 219.1 on the static boot. The seed changes where you start, not where you end.

**H2 — a faster swap queue (FAIL, closed).**

| seed boot | agent → JP/DE T90 | JP/DE → agent T90 | churn | converged C1 JP/DE / agent |
|---|---:|---:|---:|---:|
| B0 shipped (drain 64) | **8.7 s** | **14.0 s** | 95 /min | 189.6 / 218.9 |
| B1 2× (drain 32) | 9.3 s | 14.0 s | 103 /min | 189.6 / 218.2 |
| B2 4× (drain 16) | 10.0 s | 14.0 s | 112 /min | 189.5 / 217.9 |

T90 is the cold share in 2 s windows: the time to clear 90 % of the peak → converged excess, pooled over 3 reps. The
static boot's B0 read 11.0 / 14.0 s. Converged cold share is 0.3–0.5 % per shift, and cold share is ≤ 1 % at 8.7 s
and 14.0 s.

No cell halved T90 (ratios ×1.00–1.15 vs the same-boot B0). **Why:** the seed map collapses the shift peak from
**25.4 % to 5.7 %** cold share (agent → JP/DE), and from 6.0 % to 3.3 % the other way. The swap queue never saturates,
so a faster queue has nothing to drain. The shipped policy stays.

**Swap cost, measured for the first time.**
- **Per swap:** a host step that applies swaps runs **~1.9× the p95** of a swap-free step on every cell, including the
  shipped-policy control. That is 1.3–2.4 ms of excess per applied swap, of which the CUDA row copy is ~0.26 ms.
- **Whole seed boot** ([`p99_overall.py`](results/2026-10-24-v26-seeded-adaptive/scripts/p99_overall.py)): **2.53 %** of 267,226 host steps carried
  swaps. p99 host step was **22.17 ms**, against 19.52 ms for swap-free steps alone, under shift-heavy test traffic.
- These are host-step (verify-window) intervals, not client inter-token gaps.

**Quality.**
- Gauntlet **12/12** on the fixed seed map, and **12/12 live under adaptation** on the promoted lane.
- Greedy parity **49/50**, static vs seed map cross-boot. The miss is `agent-tool-1`, the documented noise prompt
  that also flips within one map.
- 0 stale swaps and 0 server errors in every cell.
- From the 2026-10-23 bundle: the restore round trip (a 15,382-swap campaign then `restore_map` returns to fixed-map
  parity, 49/50).
- **Not re-run on v26:** BFCL, GSM8K, teacher-forced Δlogprob, and any 128K or 1M request.

**Correction carried in from Part 1.** Convergence is **not 60–90 s**. That number was a windowing artifact: 30 s
medians over prompts whose converged speeds differ by up to 19 %. Re-analysis of the 2026-10-23 telemetry gives
cold share < 1 % in **9–16 s** and each prompt at ~90 % of converged speed in ~12–20 s. AR6 measures T90 at
8.7 / 14.0 s on the seed boot. The correction is live on
[Part 1](https://al-engr.com/gb300-adaptive-expert-residency.html).

**Against the sidecar sibling.** On a box with the RTX PRO 6000, the sidecar lane is still the faster one; see
[its README](../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/README.md). v26 is the single-GPU answer.

## 2026-10-23 round — adaptive expert residency (hook v26ar.1): released as EXPERIMENTAL, not promoted (superseded by v26 above)

v23's 295-hot / 89-cold split per layer was fixed at boot (`rowmap-static-v1.json`, fitted on agent traffic). Hook v26ar.1 makes the split follow live traffic. **Same image, weights, flags, and starting map as v23**; only the hook directory and its env change. Write-up: [We Built the Counter](https://al-engr.com/gb300-adaptive-expert-residency.html). Receipts, hook, diffs, and harness: [`results/2026-10-23-ar-adaptive-residency/`](results/2026-10-23-ar-adaptive-residency/).

How it works:
- **Count:** one masked `tl.atomic_add(..., sem="relaxed")` per routed pick, inside the existing split kernel, into a fixed `int32 [40, 384]` table. No new launch, no new graph node.
- **Decide:** a side thread reads the table every 64 steps and keeps an EWMA (500-step half-life). It picks swaps of the hottest cold expert vs the coldest hot one (1.5× margin, ≤ 2 per layer per read).
- **Apply:** the September swap path runs them between steps (≤ 8 per step, ~150 µs each), with a stale-decision guard.
- **Control:** a JSON ctrl file sets freeze, count, budgets, and `restore_map`.

| C1 unless noted, median after 60 s | v23 map (frozen) | adaptive | Δ | cold share |
|---|---:|---:|---:|---|
| Japanese/German essays (2 boots) | 89.95 | 189.4 / 189.8 | +111 % | 28.3 % → 0.3 % |
| agent English | 195.65 | 218.45 / 219.29 | +12 % | 2.17 % → 0.45 % |
| mixed, C1 aggregate | 133.1 | 194.1 | +46 % | 17.2 % → 0.72 % |
| mixed, C4 aggregate | 236.6 | 431.0 | +82 % | 16.6 % → 0.67 % |

- **Cost:** counter 2–6 µs/step (relaxed; acq_rel was 25–78 µs). Same-boot A/B over 12 pairs: −0.21 % ± 0.36 %. KV unchanged at 2,495,694 tokens. TTFT 2k/8k/31k: 0.19 / 0.46 / 1.38–1.50 s.
- **Fidelity:** unit tests 6/6. The 50-prompt greedy parity set matches the fixed-map capture within lane noise (49/50, the same noisy `agent-tool-1` prompt), including after a static → 15,382-swap → restore round trip. With a different hot set, one extra near-tie prompt flipped; we don't claim bit-identity across maps.
- **Not done before promotion:** soak (p99 on swap steps), gauntlet, BFCL, long-context check.
- **Against the sidecar sibling, same workloads, same night:** the sidecar lane is still faster on this box: mixed C1 208.9 vs 194.1, C4 560.4 vs 431.0 (held-out prompts), JP/DE C1 186 vs 189, agent 244 vs 219. Adaptive v23 is the better answer when you have **only** the GB300. See the sibling's 2026-10-23 round.
- **Card:** [`card-v26ar.png`](card-v26ar.png) (base v23 card: [`card.png`](card.png)).
- **Launch (experimental):** the v23 launch line with the hook directory swapped for `results/2026-10-23-ar-adaptive-residency/hook/` (plus the unchanged v19 files) and the env from `harness/launch_ar.sh`. It boots frozen (counting only); `touch $RUN/GO`, then `echo '{"freeze": false, "baseline": true}' > $RUN/ctrl.json`. Freeze again or `{"restore_map": true}` to return to the boot map.

## 2026-10-18 round — v23 promoted, nightly config C passes its gates (not promoted), P1 fails

Same box, same 1017 weights, every verdict from a bracketed window against the live lane. Bundles:
[`results/2026-10-18-v23-pinexact-host/`](results/2026-10-18-v23-pinexact-host/README.md), [`results/2026-10-18-p2-nightly-39983a5e-gate/`](results/2026-10-18-p2-nightly-39983a5e-gate/README.md), [`results/2026-10-18-p1-prefill-graphs-FAIL/`](results/2026-10-18-p1-prefill-graphs-FAIL/README.md).

### v23 Pin Exact — promoted 10:21 CDT

`launch_v22.sh` + `-e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024`. torch 2.13 rounds every pinned host
allocation up to a power of two, so the two 94.4 GiB Engram tables had been pinned as 2 × 128 GiB. Above the cap they
are now pinned at exact size. Nothing on the GPU changes: same image, kernels, autotune hash (`62426808`, Loaded 189),
KV pool and expert split.

| | v22 (controls) | **v23** |
|---|---|---|
| pinned host (EngineCore `/dev/zero` maps, smaps Size sum) | 327.96 GiB | **252.79 GiB (−75.17)** |
| MemAvailable steady / after drop_caches | 118–124 / 131–133 GiB | 193–196 / 206 GiB |
| hook guard `host_avail` at layer-9 rehome (floor 10 GiB) | 11.3–12.1 GiB | **197.7 GiB** |
| HBM experts / Grace-pinned experts · KV tokens | 206.61 / 62.33 GiB · 2,504,891 | same · same |
| RT C1 / C4 / C16 real text (Card S0, bracketed) | 186.5 / 318.7 / 522.9 · 193.5 / 335.9 / 540.2 | 190.2 / 325.1 / 523.5 (+0.1 / −0.7 / −1.5%) |
| replay r2 · gauntlet | 441 / 437 · 12/12 | 438 (−0.2%) · 12/12 |
| greedy non-tool parity (e2c_parity, 16 prompts) | — | **16/16 vs CTL1 and 16/16 vs CTL2** |
| confirm (Card S0c): knee6 r2 C8 / C6 / RT C1 (mean of 3) | 597.0 / 535.8 / 187.8 | 604.8 / 529.6 / 188.2 (+1.3 / −1.2 / +0.2%) |

The boot-guard margin is what changed in practice: the hook's 10 GiB host floor had aborted three boots on
2026-10-05/06, and it now clears by ~188 GiB. **Prose-knee C8 is bimodal on this lane** — each rep lands fast
(~620–632) or slow (~556–583) on every arm — so S0's −3.4% C8 was mode luck; C8 is now gated on ≥ 4 reps.

### Candidate (not promoted) — config C on vLLM nightly-39983a5e

`vllm/vllm-openai@sha256:a3bc7f09…2623` + hook v21 + off60 + `--kv-cache-dtype fp8_ds_mla` + `FLASHMLA_MEGA_ATTN_DSV41`
+ `--no-swa-bounded-replay` + the v23 alloc env + a managed autotune store (`vllm-cache-p2-39983a5e`, 210/210 preloaded).
Card P2f, 19:09–20:18, v23 live → C → v23 restart; bars fixed in the runner before launch. **PASSED GATES — NOT
PROMOTED (pending James).**

| gate | bar | config C vs v23 | |
|---|---|---|---|
| per-class decode wall-clock (5 classes × C1/C8, two v23 controls) | ≥ ctl mean − max(2%, 1.5 × ctl spread), every cell | 20/20 cells, +0.5 … +12.6% | PASS |
| GSM8K-200 | ≥ A − 1.5 pt | 95.5 vs 97.0 (−1.5) | **PASS, at the bar** |
| BFCL dev (600) | ≥ A − 1 pt | 92.50 vs 92.33 | PASS |
| BFCL held-out (1,311) | ≥ A − 1 pt | 77.65 vs 77.65 | PASS |
| gauntlet | 12/12 | 12/12 | PASS |
| RT C1 real text, mean of 3 | ≥ CTL − 2% | 197.8 vs 186.4 (+6.1%) | PASS |
| knee6 C8, 4 reps per arm | ≥ CTL − 2% | 656.5 vs 605.2 (+8.5%) | PASS |
| margin-weighted TF flips (Card P2, same image, replay on) | W ≤ 3.85 | 3.11, 0 confident agent/tool flips | PASS |

Not gated: TTFT 8K 0.430 vs 0.645 s · cold ~53K prefill 30.5K vs 24.0K tok/s (+27%) · KV 3,878,024 tokens (+55%) ·
knee r2 C16 1,075 vs 944 (+14%) · engine step ~6% faster at equal acceptance in every class.

**GSM8K caveat.** −1.5 pt is exactly the line, the 3 flips all go one way, and config C writes **+48% completion
tokens** on GSM8K (mean 293.5 vs 198.3; v23 restart vs live is +6%). Two of the three flipped items are long C outputs.
n = 200 cannot separate 1.5 pt from noise (p = 0.25); the full 1,319-item run per arm has not been done. Structured C1
acceptance −2.6 / −3.6 pts trips the investigation trigger (explanation not verified). The 8-prompt fixture's code
−4.7 pts, which failed P2b–P2e under the old rule, does not reproduce on 40 real prompts per class (within 0.4 pt).

How it was reached: P2 (replay on) failed warm replay r2 −10.9% — vllm#58132 SWA bounded replay makes a prefix-hit
request cost a cold one; the replay-on variant is parked (TTFT 8K 0.28 s, cold prefill ~×1.9, warm replay r2 −6…−11%).
P2b → P2e then ruled out the #56893 MXFP8 KV record, the drafter's and the target's attention kernels and MegaAttn fused
cast, and a desk check ruled out the aux-capture path (same tensor as the 1017 reference). Kept stopped as
`dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP-FAIL-P2f-PASS`.

### FAILED — Card P1, prefill CUDA graphs to 8,192 tokens

| arm | capture sizes | graph memory | KV tokens | verdict |
|---|---|---|---|---|
| v22 control | 1 … 128 | 1.29 GiB | 2,504,891 | — |
| CG8K | + 256 … 8192 | 8.21 GiB | none — boot died (KV −1.6 GiB) | **FAILED** |
| CG2K | + 256 … 2048 | 3.63 GiB | 1,420,753 (−43%; bar: loss ≤ 0.3M) | **FAILED** |

Graph memory is super-linear in the top capture size and charged to KV; the capture list is part of the FlashInfer
autotune hash (new hash, provisional boot). Reopens only with an HBM source.

### C1 headline: mean of 3, acceptance next to it, split by class

The index headline is now the mean of three real-text C1 runs with `acceptance_len` (1 + accepted / draft steps, bonus
token included), plus a per-class split (`docs/benchmarking.md`, "C1 headline rule"). v23: **188.2 tok/s, 2.99 tokens/step**
(190.8 / 189.1 / 184.8). Per class on the per-class instrument (40 real prompts per class, forced 256 tokens, decode only,
C1): prose 232.3 / 3.09 · structured 250.3 / 3.27 · code 293.4 / 4.33 · shell_ops 316.9 / 4.42 · tool_json 300.0 / 3.98
(tok/s / tokens per step). The two instruments use different prompts and output lengths (CNN/DM 8.2K in, 1,024 out vs
real Hermes contexts, 256 out), so compare within one column.

Also delivered: `longgen_cell.py` (after original-el8's `long-gen.py`), a long-generation cell; dry run 2,048 tokens at
219 tok/s decode, acceptance 3.52. The 60K-token cell has not been run.

## Measured, not adopted — Grace-only MegaMoE via el8's M3 hook, and 32 seats (2026-10-18, 23:04–01:52)

Question: can the one-Station lane improve **without** the RTX PRO 6000 sidecar? Two candidates, one bracketed window on the 1017 weights, RonanLabs' real-text harness (CNN/DM 8.2K in, 1,024 forced, nonce, 0% prefix hits) plus our knee/fixture/replay/gauntlet. Bundle: `results/2026-10-18-cardM3G-grace-only-megamoe/`.

**M3G — original-el8's M3 hook in Grace-only mode** (`MEGA_PEER=0 MEGA_COLD_TRT=1`) on `nightly-af7f9488` (pull by digest `sha256:3af6a45b…`, the tag has rotated off Docker Hub) with the vllm#58132 replay overlay: MegaMoE is constructed over the hot rows only, so DeepGEMM never sees all 384 experts (the stock `deep_gemm_mega_moe` OOM, Card F); cold rows run on the TRT-LLM MXFP4 kernel from pinned Grace — this recipe's cold path. 32 seats, k `[[1,4,5],[5,8,3],[9,32,3]]` with probabilistic drafts, `MEGA_LL_GEMM=1`.

| same window, controls = CTL1/CTL2 mean | v22 | M3G, el8 public map 285/99 | **M3G, our map 295/89** | V22S32 confirm |
|---|--:|--:|--:|--:|
| real-text C1 / C4 / C16 | 188.5 / 333 / 538 | 147 / 268 / 459 | 176 / 341 / **654** | 186 / 322 / 542 |
| C32 output tok/s | 511 (24 seats, 8 queued) | 535 | **739** | 640 |
| TTFT 8K | 0.65 s | 0.33 s | **0.29 s** | 0.64 s |
| knee C1 / C8 / C16 | 192 / 619 / 938 | 164 / 512 / 691 | 180 / 672 / 934 | 200 / — / — |
| replay n=4 (agent transcripts) | 415 | 329 | 405 | 418 |
| KV at 1M | 2.50M | 6.05M | 2.49M | — |
| gauntlet | 12/12 | 12/12 | 12/12 | — |

- **M3G with our map fails the C1 bar (−6%) and passes everything else** (C16 +22%, TTFT 2.3×, gauntlet, replay within noise). Not promoted: the lane's traffic is C1–C8 agent turns. It is kept stopped as the measured **sidecar base** for the RTX PRO 6000 (same image and hook; `MEGA_PEER=2`).
- **The map beat the kernel.** Same hook, 285 → 295 hot experts per layer: +20% C1, +42% C16. el8 ships this recipe's `rowmap-static-v1` as his default.
- **MegaMoE loses to the FlashInfer TRT-LLM hot kernel at M≈6** (C1 fixture prose −9%, tool_json −13%) and wins at large T (prefill) and at C16+ once the k=3 band and probabilistic drafts raise acceptance (2.66 vs 1.80 per step).
- **32 seats on v22** passed its pre-registered bar only at the edge on a bracketed confirm (C1 −2.4%, C4 −3.0%, C32 output +24%, C16 TTFT worse); the first boot's knee C1 +10% was autotune luck. Not promoted; the seat count moves with the sidecar.
- Upstream facts that changed the notes (Brief B, GitHub record): vllm#58132 is **not** in af7f9488 or v0.31.0 (it is in the v0.32.0 tag and nightly-39983a5e); `#55095` is a full-decode-graph fallback fix, **not** adaptive verification (`#47808`, `#52228`); `#61509` (adaptive cudagraph memory) is still open; upstream MegaMoE `#60096` allocates all experts in HBM at construction — still no partial residency; `#61371` (nightly ≥ 10-15) changes the V4.1 KV layout. SGLang re-checked: no in-place pinned-host expert read on main; stays parked.

Credits: original-el8 (M3 hook, Apache-2.0), RonanLabsAI (real-text harness and the Grace-only control numbers that prompted the card), 0xSero (bench methodology review).

## Release notes — v22: k=5 band widened to 8 live sequences (2026-10-17, 15:27–16:55)

`num_speculative_tokens_per_batch_size` `[[1,4,5],[5,24,1]]` → **`[[1,8,5],[9,24,1]]`**; nothing else changes (same image, hook, offload, `num_speculative_tokens=5` → same autotune hash, 8-minute boot). Trigger: porting 0xSero's 4× RTX PRO 6000 grid to the lane showed the Station at 0.56× his total decode at C6–C8, and a flat-k=5 arm recovered most of it — the k=1 band was starting too early for high-acceptance output.

| instrument | CTL (v21) | **v22** | CTL (v21) | read |
|---|---|---|---|---|
| Sero grid 8192-in, C6 / C8 total tok/s | 345 / — | **542 / 602** | — | +50–58% (Card T; CTL from the same hour) |
| prose knee C1 / C4 / C8 / C16 | 191.6 / 421.4 / 667.2 / 937.2 | 191.9 / 421.2 / **621.2** / 933.2 | — | C8 −6.9%, rest flat (Card T) |
| replay, 4 workers, r1 / r2 | 426 / 414 | 293* / 423 | 280* / 418 | flat; *cold r1 on fresh boots (Card U) |
| replay, 6 workers | 415 / 443 | **461 / 479** | 402 / 411 | +8…+17% (Card U) |
| replay, 8 workers | 456 / 467 | **473 / 484** | 461 / 453 | +2…+7% (Card U) |

A 4-worker replay keeps ≤4 sequences live, where both schedules run k=5, so it cannot see a 4→8 boundary — the reviewer (Opus 5.5) caught that and Card U is the instrument that can. Both arms hash-hit; hook re-homed 206.61 / 62.33 GiB on each. Promoted 16:55 on James's go; verified end-to-end on Milo (`custom[dsv41]`), the Forge guest gateway (`:4000`, `deepseek-v4.1-flash`) and Echo/Alyx. Prometheus now scrapes the lane (`dsv41-lane`, 15 s) so the next schedule question is asked against a real concurrency histogram instead of "1–4 typical". Receipts: `~/dsv41/1017/results-cardT/`, `results-cardU/`; ledger 16:55.

## Release notes — DeepSeek-V4.1-Flash-1017 on the v21 config (2026-10-17, 07:15–09:55)

Weights-only update (`1bae0e10`; config.json, shard index and tokenizer byte-identical to `df42c109`). Same launch, same hook, same autotune set, same rowmap. Promoted to `:30006` as `dsv41-vllm-1017-v21cfg-BOUND-REF`; v21 kept as `-RETIRED-REF`.

Same-window CTL1 → CAND → CTL2 (knee r2, agg tok/s; gauntlet 12/12 all arms; CTL1↔CTL2 greedy identical 18/18):

| | C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|---|
| df42c109 (CTL1) | 182.7 | 286.3 | 433.1 | 685.6 | 841.2 | 980.9 |
| **1017 (CAND)** | **192.6** | 288.9 | 423.9 | 672.6 | 850.9 | 946.1 |
| df42c109 (CTL2) | 182.6 | 286.3 | 434.2 | 686.0 | 852.9 | 984.9 |
| Δ vs bracket | **+5.4%** | +0.9% | −2.2% | −1.9% | +0.5% | −3.7% |

Drafter acceptance moved (retrained DSpark head): prose 34.9→25.2%, structured 52.5→48.9, code 66.1→**69.6**, shell 90.8→**80.0**, tool_json 87.8→**77.3**. C1 prose got *faster* anyway — fewer accepted drafts per step means fewer unique experts fetched per step. Replay on identical transcripts emits 14 tool turns vs 5–6 (more eager tool use). Quality: GSM8K-200 96.5% vs 97.5% (McNemar p=0.5, n.s.), needle-checksum 500/500, completion tokens −13.6% (GSM8K) / −24% (needle).

Schedule re-check (Card C): `[[1,2,5],[3,24,1]]` wins the C4 prose knee (+12%) and loses real agent replay on both reps (−8…−23%) — the v13 mechanism, unchanged by the new drafter; `[[1,4,3],[5,24,1]]` is −42% C1 (a k<max band inside a k=5 schedule pays the full verify shape). **Keep `[[1,4,5],[5,24,1]]`.**

Promotion waiver: the "acceptance within 2 pts" rule was written for engine changes on fixed weights; a retrained drafter moving acceptance is a checkpoint property (same image, same kernels, controls bit-identical). Waived explicitly in the box ledger.

Card F (same day): vLLM 0.31.0 `deep_gemm_mega_moe` is a hard no on one Station — `DeepseekV4MegaMoEExperts` materializes a full interleaved GPU copy of all experts at first forward (`deep_gemm.transform_weights_for_mega_moe`) and drops the loader params, so UVA offload cannot reduce HBM; 269 GiB > 250. Receipts `results/2026-10-17-ds1017-cardA-cardC-cardF/`, quality `results/2026-10-17-ds1017-cardD/`.

### Client settings (added 2026-10-17 — a Desktop session failed on this)

The V4.1 chat template validates `reasoning_effort`. Measured against this lane, one request per value: **`none`, `low`, `high`, `xhigh`, `max` → 200; `medium` → HTTP 400** (`DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, or an integer within [1, 100] in chat_template_kwargs`). Integers 1–100 are accepted only inside `extra_body.chat_template_kwargs`; a top-level integer fails pydantic validation. `medium` is the OpenAI default and the Hermes Desktop composer default, so a fresh client hits this on its first turn.

Hermes: `agent.reasoning_overrides: {dsv41-flash-uva: low}` covers the gateway, cron and CLI, but **not** a Desktop session — the composer's effort selector sends a per-session override that outranks the config. Pick `low` or `high` in the composer for this model. `limits:`/`clients:` in `recipe.yaml` carry the same note.

## Standard bench T2/T3 on v21 (2026-10-06 evening)

Bundle: [`results/2026-10-06-lib-t2t3-v21/`](results/2026-10-06-lib-t2t3-v21/README.md). Fresh host boot (the reboot recovered the PIN_HOT boot margin: 12.47 GiB at the layer-9 check).

| Test | v21 |
|---|---|
| tool-eval-bench (leaderboard settings) | **92/100**: 81 pass · 7 partial · 4 fail, including **cross-tenant disclosure (TC-92)** and a **false "invoice paid" after a partial failure (TC-89)** |
| GSM8K 1,319 | 96.9 % (same-config paired re-run vs the 200-item baseline: p = 0.5, this lane's noise floor) |
| MMLU-Pro 1,000 | 87.1 % |
| GPQA-Diamond 198 | 89.9 % |
| needle-checksum | 500/500 EXACT |
| Decode, filler, agg tok/s | C1 184–214 across 0–128K · C8 576–594 · C24 995–1022 (0–64K) |

The bench's burst cells flag repetition loops in 16 of 29 cells. That is an artifact of the bench forcing generation past EOS (`ignore_eos`), not an answer-time loop. Details are in the bundle; a `--respect-eos` re-run is pending.

## Standard bench baseline: llm-inference-bench on v21 (2026-10-06)

Bundle: [`results/2026-10-05-lib-t1-v21/`](results/2026-10-05-lib-t1-v21/README.md). Method: [`docs/benchmarking.md`](../../../docs/benchmarking.md).

This is the first run of the pinned third-party bench ([llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) v0.7.7 @ `c71ec1f2`) against the unchanged v21 container. It does not replace the 182.8 headline. The bench decodes its own synthetic filler, and DSpark acceptance depends on content. What it adds is a repeatable lane baseline and a quality reference file for future A/B runs.

| ctx \ conc (agg tok/s) | 1 | 4 | 8 |
|---|---|---|---|
| 0 | 203.5 | 344.5 | 561.0 |
| 32K | 173.7 | 367.7 | 572.1 |
| 128K | 190.2 | 376.3 | 558.0 |

- **Prefill:** 15.4K tok/s at 8–32K and 18.5K at 64–128K (128K prompt → 6.93 s TTFT).
- **needle-checksum:** EXACT 500/500 (identical greedy requests through a tool call). There is no numerics drift on this boot.
- **GSM8K-200:** 97.5 % (195/200, Wilson 94.3–98.9). This file is the `--compare-baseline` reference for paired McNemar tests on the next engine or kernel change.
- **Boot margin warning:** v21 aborted three times at the hook's 10 GiB host-memory floor before this boot passed at 10.29 GiB. Details are in the bundle and `limits`.

## Round 12 — the step profile after residency, huge pages closed, and one fused kernel worth +6 % (2026-09-27, 05:00–08:20)

Bundle: [`results/2026-09-27-v21-fused-split/`](results/2026-09-27-v21-fused-split/README.md).

**Huge pages, closed in 30 minutes without a boot.** A GPU-side microbench of pinned Grace memory — random 256 B gathers (Engram-shaped) and 17.9 MiB streams (expert-shaped) — reads the same on 64 K pages, on THP-madvise anon and on 100 % hugetlbfs 512 MiB pages: ~118 GB/s / ~355 GB/s, all within ±1 % (HBM control 973 GB/s / 1.8 TB/s). C2C is bandwidth-bound, not translation-bound; `cudaHostAlloc` regions ignore `shmem_enabled` entirely. That also explains the September-11 THP non-result.

**First profile of the v18 step.** Torch profiler on the exact v18 config: a C1 prose decode step is 14.6 ms at 101 % GPU-busy with a 0.7 ms inter-step gap — no CPU bubble. Expert GEMM 5.0 ms (34 %; 56 launches at 53 + 28 µs on M≈6 — tile-quantized, ~4× the bandwidth floor), dense GEMM 4.2 (29 %), **small kernels 2.7 ms over ~1,300 launches (18 %)**, attention 1.2 (8 %), mHC 1.0, Engram 0.04. The v15 hook's two-call hot/cold split is nearly free (cold-side launches are 5 µs no-ops), but its `split_ids()` was six torch ops per layer — ~400 launches of the swarm.

**v21.** One Triton kernel does the same integer remap. Bit-exact vs the torch path at every T; then a same-window CTL → CAND → CTL on fresh boots, same autotune set:

| | C1 | C2 | C4 | C8 | C12 | C16 | replay n=4 |
|---|--:|--:|--:|--:|--:|--:|---|
| v18 hook (CTL1) | 171.8 | 274.3 | 414.9 | 656.0 | 809.3 | 957.0 | 343 / 450 |
| **v21 hook** | **182.8** | **287.0** | **433.7** | **686.8** | **845.0** | **992.8** | 335 / 463 |
| v18 hook (CTL2) | 172.5 | 272.8 | 415.8 | 657.1 | 813.4 | 952.4 | 332 / 470 |

+6.2 % C1, +4.0–4.9 % everywhere else, replay a wash, acceptance unchanged; non-tool greedy 16/16 identical against both controls (the one differing prompt is a tool-call prompt of the documented jitter class). Promoted 08:12 CDT; v18 kept stopped as rollback. **Note:** this was measured against v18 because the lane had been on v18 since v20 was stopped on 2026-09-21 20:57 CDT (no ledger entry). v20 + hook v21 should compose and is unmeasured; held-out BFCL on v21 is pending.

## Round 11c — the "unexplained" C8 loss was the capture-size list, and it is a trade worth keeping (2026-09-21, 14:04–15:30)

One pair on the v20 image: `--cudagraph-capture-sizes` omitted (engine default) vs the 15-size list, same hook, same window. Receipts: [`receipts/cardG/`](results/2026-09-21-v20-promotion/receipts/cardG/).

| | default capture | 15-size list (v20) |
|---|---|---|
| C1 / C4 / C8 / C16 | 173 / **443** / **720** / **1007** | **180** / 400 / 657 / 969 |
| KV pool | 1.86M | **2.55M** |
| fund C16 warm agent p95 | 1.73 s | 1.68 s |
| fund C24 warm agent p95 / mean | 9.5 s / 2.7 s | **2.4 s / 0.98 s** |
| fund C24 aggregate | 249 | **283** |

So the Sept-18 note ("24-slot profiles cost ~6% at C8, unexplained") is closed: the list costs ~10% on the 4–8 stream knee and buys +4% C1, +0.7M KV tokens and a 24-seat lane that stays warm. That is the right trade for this lane's job (the 35-seat fund harness). A deployment that lives at 4–8 streams and never seats 24 should drop the flag and take the knee. Also this window: the GLM Flash draft `bf582e4e` A/B on `:30001` — see that recipe; no change, pin kept.

## Round 11b — the bisect finishes on one PR, and v20 is promoted (2026-09-21, 07:38–12:30)

Two more nightlies split the `af1c0149` set, then the v20 window (r1 → v18 → r1, loaded autotune), then the first fill of the agent claim card. Bundle: [`results/2026-09-21-v20-promotion/`](results/2026-09-21-v20-promotion/).

| nightly | adds | agent-tool flips | held-out | accept tool_json / shell / prose |
|---|---|--:|--:|---|
| `2671fedf` 09-13 | — | 1.77% | 0.00% | 0.860 / 0.908 / 0.349 |
| `dc36fcce` 09-14 | #56512 Engram prefetch, #56464 DeepSelect (opt-in) | **1.77%** | **0.00%** | 0.860 / 0.908 / 0.349 |
| `cd10ed6f` 09-15 | **#56633 fold mHC post into delayed pre projection** | **12.66%** | **9.00%** | 0.878 / **0.829** / 0.319 |
| `af1c0149` 09-16 | #56962 Mega-mHC | 11.73% | 8.72% | **0.757 / 0.760 / 0.260** |

`dc36fcce` is bit-identical to `2671fedf` on the teacher-forced corpus: Engram prefetch and DeepSelect are exonerated. **The drift enters with #56633**, one PR, same box, same weights, T=0 logprobs; Mega-mHC on top is what costs DSpark acceptance on tool-JSON. Filed upstream as a single-PR finding (see the ledger).

**v20 window**, same morning, loaded autotune set `a34f9ad4`: C1 **180.9 / 171.8 / 180.3** (v20a / v18 / v20b — +5.3% / +4.9%, pair mean +5.1% against the ≥5% bar, met on the mean and stated so), C16 979 / 952 / 965, fund C24 warm agent p95 2.83 / 3.36 / 2.42 s, cold 120K p50 8.4 / 8.4 / 8.6 s, tools 64/64 on all three. **Held-out sibling** (BFCL live_simple + live_multiple, 1,311, frozen this morning, run once): v20 **78.8%** vs v18 **78.9%** — pass against ≥ v18−1. Dev suite on the same boots 92.7% vs 93.0% (v20 was 93.5% the night before; ±1 pt is run-to-run on 600). Cost per 1,000 solved held-out tasks at $0.15/kWh: $0.008 vs ~$0.007 (v18 W estimated — its power sample was lost to a script bug; labelled). **Promoted 12:30 CDT; v18 is the rollback.** The promotion is to an eight-day-old nightly on purpose: every later nightly inherits #56633.

## Round 11 — where the drift enters, and a v20 candidate (2026-09-21, 05:30–06:55)

Three nightlies between the 0909 image and the reverted v19, hook on, v19 flags, teacher-forced split vs the no-hook capture, acceptance, knee. Bundle: [`results/2026-09-21-kernel-bisect/`](results/2026-09-21-kernel-bisect/README.md).

| image | main date | adds | agent-doc flips | accept tool_json / shell / prose | C1 |
|---|---|---|--:|---|--:|
| `0909` (v18) | 09-10 | — | **0.56%** | 0.878 / 0.908 / 0.349 | 172 |
| **`2671fedf`** | 09-13 | model support on main, no perf kernels | **1.77%** | **0.860 / 0.908 / 0.349** | **189 live / 180 loaded** |
| `af1c0149` | 09-16 | #56464 DeepSelect, #56962 Mega-mHC, #56512 Engram prefetch | **11.73%** | 0.757 / 0.760 / 0.260 | 194 |
| `0bfc7a15` | 09-17 | #56935 mega-attn + NVFP4 KV, #56568/#57204 MegaMoE | 10.99% | 0.838 / 0.829 / 0.336 | 183 |
| `dee37d89` (v19) | 09-18 | #56266 Mega-Gate | 11.45% | 0.838 / 0.829 / 0.336 | 183 |

**The drift enters with the `af1c0149` PR set** — DeepSelect top-k for the sparse indexer, Mega-mHC, Engram prefetch; this bundle does not separate the three (Round 11b does: it is #56633). The MegaMoE/mega-attn rung then recovers part of the acceptance without touching the flips. **`2671fedf` is the v20 candidate**: v18's acceptance on every class, BFCL **93.5%** (v18 93.3), +5–10% C1, and an order of magnitude less drift than v19 — it passes the promotion rule as written. Not promoted: it needs a same-window v18 pair, the fund harness at C16/C24 and a pinned autotune set in a scheduled window. `0bfc7a15` and `dee37d89` share the `ddf01704` autotune hash, so Mega-Gate does not change MoE shapes.

## Round 10 — the tool-call gate, and why v19 came back out (2026-09-20 evening)

v19 (nightly `dee37d89` + hook + off54 + `fp8_ds_mla`) went live at 12:11 on its speed bars. A Grok review that afternoon asked the question the bars did not: *is tool calling still correct?* The per-class teacher-forced split said the logit drift concentrated on tool-shaped text (9.6% top-1 flips vs 0.9% for the v18 hook), and the promotion rule was rewritten the same evening: **BFCL exact-match ≥ v18 − 1 pt AND DSpark tool-JSON/shell acceptance within 2 pt of v18.** Replay tok/s is a speed bar, never a correctness bar. Bundle: [`results/2026-09-21-toolcall-gate-49435/`](results/2026-09-21-toolcall-gate-49435/README.md) (runners, `bfcl_gate.py` grader, `tf_split.py`, every JSON).

| | v18 | v19 | nightly `d05da62e` + #49435 at v19 flags |
|---|--:|--:|--:|
| BFCL v4 simple_python + multiple (600, T=0, our AST grader) | **560 = 93.3%** | 559 = 93.2% | 564 = 94.0% |
| failure kinds | value 36 / name 3 / no-call 1 | value 37 / name 2 / no-call 2 | — |
| DSpark accept tool_json / shell | **0.878 / 0.908** | 0.838 / 0.829 | 0.806 / 0.829 |
| agent-doc top-1 flips vs no-hook (TF, 1,074 pos) | **0.56%** | 11.45% | 11.45% |
| GPQA-Diamond, 64k budget | 172/198 = 86.9% | 173/198 = 87.4% | — |
| C1 / C8 / C16 prose | 172 / 655 / 950 | 183 / 667 / 995 | 193 / 711 / 1037 |

**Correctness is a wash; acceptance fails; the cause is the image.** BFCL is within 0.2 pt with identical failure kinds (zero JSON-parse, extra-arg or missing-arg failures on either side — the unique misses are value-normalisation and optional-arg choices). But the drafter agrees with the target 4–8 pt less often on exactly the text an agent lane emits, and that is the same fact the TF split was reporting. The cheapest physical candidate — vLLM #49435's `fp8_ds_mla` writer-scale fix, merged after our pin — was tested in the same window: a boot on nightly `d05da62e` (which carries it) at the v19 flags leaves the agent-doc flips at **11.45%, unchanged to two decimals**. The drift is in the DSV4.1 kernels that landed between the 0909 image and `dee37d89` (#56935 mega-attention/NVFP4 KV, #56464 DeepSelect, #56568/#57204 MegaMoE), not in the KV cache path. That boot also selected `FLASHMLA_MEGA_ATTN_DSV41` — the backend #56625's own PR text calls "broken upstream for this checkpoint" — and it live-tuned in **16 min**, not 74, on the same FlashInfer.

**Reverted 21:42 CDT** per James's pre-authorisation: `:30006` is `dsv41-vllm-v18-cgsizes-BOUND-REF` again (`fp8_ds_mla`, hook 206.61/62.33 GiB, KV 2.50M, autotune hit); v19 and the `d05da62e` candidate are stopped-and-kept. The nightly's speed is real and not adopted. The equal-budget GPQA pair (v18 at 64k, the missing half from the fidelity round) is a wash. Next lever, if wanted: bisect the kernel PR set one image per boot — at 16-min tunes it is an evening, not a week.

**Lesson for the recipe:** a speed win on a spec-decode lane that comes with lower acceptance on a text class is not a win on that class — acceptance *is* the argmax-agreement measurement, and it was on the fixture the whole time. The card now carries a BFCL row and the per-class TF split; a corpus-wide fidelity average never goes on it alone again.

## Round 8 — overnight k-schedule / nightly / adaptive, and the open T3b finding (2026-09-19)

Five boots, one axis each, every verdict from a same-window pair against the live v18 (`dsv41-vllm-v18-cgsizes-BOUND-REF`). Bundle: [`results/2026-09-19-overnight-ksched-nightly-adaptive/`](results/2026-09-19-overnight-ksched-nightly-adaptive/) (plan, briefs, runners, every window's JSON, the two FlashInfer autotune files). Nothing promoted; v18 stays the reference.

| test | axis | C1 | C8 | C16 (paired) | replay n=4 | verdict |
|---|---|--:|--:|--:|--:|---|
| v18ctl (×5 windows) | — | 171.7–172.5 | 642–666 | 942–954 | 320–341 / 416–475 | reference |
| T1 `[[1,4,5],[5,12,3],[13,24,1]]` | spec K mid-band k=3 | −0.7% | **−6.2%** | +0.1% | −3.8% | **FAIL** — k=3 is verify tax at C8; replay's 4 workers never leave the k=5 band |
| T2 `[[1,8,5],[9,24,3]]` | spec K | −0.3% | **−12.9%** | −6.1% | −2.7% | **FAIL** — k-schedule lever closed on this lane |
| T3 nightly `dee37d89`, hook **off** | image | 97.1 (+9.5% vs v14 positional 88.7) | 312 | 352–426 | 166 / 230 | binds clean; ≈½ of v18 without the hook — not a lane |
| T4 E4b D1 counter, unfrozen | adaptive placement | −3.0% | **−2.9%** (bar −1.5%) | −1.0% | +10.8% (±20% instrument) | **FAIL** — 12 swaps, all in the first drain, none after; adaptive-on-this-hook closed |
| **T3b** nightly + hook, off54, **live autotune** (two boots) | image + hook | **183.6 / 195.7** (+6.9% / +13.7%) | 701–733 (+8–10%) | 1014–1031 (**+21%**) | 355–374 / 468–490 (+14%) | **resolved by T3b5 below** |
| T3b2 / T3b3 same lane, **cache loaded** (docker start / fresh run with cache) | — | 172.1 / 172.3 (0%) | 677 (+4%) | 851–988 | 277–285 / 389–392 (−9%) | = v18, with a **C4 −15% / C12 −17%** hole |

What T3b established: the nightly's merged DSV4.1 kernels plus the v15 hook are worth +7–14% C1 and +21% C16 over v18 — **when FlashInfer tunes live in-process**. Loading a saved autotune cache on `0.6.18.post1` reproduces the tune's greedy tokens exactly (16/16 text prompts) but not its speed. Two live tunes select different configs (27/42 MoE, 58/147 GEMM entries) and different tokens; greedy parity vs the 0909 image is 2/18 by construction (different kernels), so cross-image parity bars do not apply.

**T3b5 resolved it the same evening** ([`T3b5/`](results/2026-09-19-overnight-ksched-nightly-adaptive/T3b5/README.md)): the 196-tok/s set pinned as the cache, two fresh boots (both hash-hit), same-window v18 control between them:

| | C1 (r1/r2/knee6) | C4 | C8 | C16 (r2) | replay n=4 | parity |
|---|--:|--:|--:|--:|--:|---|
| T3b5 (pinned set `5440f02d…`) | **187.8 / 188.2 / 188.0** | 425 | 677 / 666 | 983 | 312 / 460 | 17/18 vs T3b5b · 17/18 vs T3b4 (the tune that made it) · 4/18 vs the slow set · 2/18 vs 0909 |
| v18ctl6 | 171.5 / 171.6 / 171.5 | 414 | 657 / 657 | 952 | 331 / 437 | — |
| T3b5b (second boot, same set) | (cold 155.3) / 187.8 / 187.9 | 423 | 668 / 673 | 977 | 286 / 467 | |

**Neither fork as written.** A pinned good set gives a stable **+9.6% C1** — not the 196 of the live tune, not the 172 of the slow set — and the C4/C12 hole is gone. So both mechanisms are real: *which* configs got saved is worth 172→188 (both loaded), and ~4% exists only in a live tune. Content classes: code +3.6%, but prose −10.5% / tool_json −10% / structured −8% on the agent fixture, because the nightly accepts fewer drafts on every boot (prose 0.288 vs 0.349, tool_json 0.81 vs 0.88). **Not promoted**: `:30006` is a tool-heavy lane and a prose-knee win does not pay for that. What it does establish for the recipe: on FlashInfer `0.6.18.post1` the autotune file is a pinnable artifact (record its sha256) and the ~4% live-vs-loaded gap plus the 9% between two saved sets is a FlashInfer autotuner finding to file upstream with both JSONs. First windows on this lane with power/clock receipts: cap 1300 W, throttle `0x0` throughout, 207 W idle → 566–609 W under the knee.

**T3b6 (same night) found the acceptance cause** ([`T3b6/`](results/2026-09-19-overnight-ksched-nightly-adaptive/T3b6/README.md)): the nightly logs `Using DeepSeek's nvfp4_ds_mla KV cache format` where 0909 logs `fp8_ds_mla` — #56935's SM100 default, also why its KV pool read 4.53M tokens. One flag, `--kv-cache-dtype fp8_ds_mla`, same-window T3b6 → v18ctl7 → T3b6b:

| | C1 | C4 | C8 (r2) | C16 (r2) | replay n=4 | prose / structured / tool_json accept |
|---|--:|--:|--:|--:|--:|---|
| v18ctl7 | 172.1 | 417 | 665 | 966 | 319 / 434 | 0.349 / 0.525 / 0.878 |
| **T3b6** live tune | **193.7** (+12.6%) | 445 | 742 | 1052 | 380 / 484 | **0.336 / 0.492 / 0.838** |
| **T3b6b** cache loaded | **183.3** (+6.5%) | 417 | 630 | 986 | 316 / 485 | same |

Most of the acceptance came back (the residual 1–4 points are the nightly's own numerics; parity vs 0909 is 2/18 on every nightly boot), and the speed *grew* — the +7–14% is the merged DSV4.1 kernels, not the fp4 KV. The KV dtype changes the autotune hash, so T3b6 tuned live (new set `ddf01704…`, sha256 in receipts) and T3b6b loaded it: **−5.4% with identical tokens, the third live-vs-loaded observation** (−4 / −6 / −5%). Fixture: tool_json −6% still fails the −3% bar; the rest is within bar on the live boot and slips to −4…−6% on the loaded one. **v19 candidate, not promoted** — needs a second loaded-config pair, the fund harness at C16/C24, and a ruling on tool_json for a tool-heavy lane. Reference stays v18.

**v19 second pair + fund harness (2026-09-20)** ([`results/2026-09-20-v19-pair-async/`](results/2026-09-20-v19-pair-async/README.md)): two fresh boots of the loaded v19 configuration (nightly + hook + off54 + fp8 KV + pinned `ddf01704…` set) land at **183.2 / 183.1 tok/s C1** (four runs within 0.1; +6.6% vs same-window v18) and are **token-identical to each other (18/18)**. On the fund harness: C24 aggregate +8% (306–310 vs 285), C24 warm-agent p50 0.55 vs 0.60 s, cold 120K p95 8.3 vs 8.4 s, tools 64/64. One miss: v19b's C16 warm-agent p95 2.56 s (bar 2.28; v19a 1.68). C8 knee −5% on every loaded nightly boot, unexplained. **Candidate holds; not promoted** — needs one more C16 fund window and a ruling on the fixture's tool_json −6%. Same session: **`--async-scheduling` on v18 is a wash** (C1 −0.4%, C8 −0.3%, C16 −0.6%, acceptance identical) — closed at v18 as at v12.

**T3c (2026-09-20) measured the flag-only row** ([`T3c/`](results/2026-09-19-overnight-ksched-nightly-adaptive/T3c/README.md)) for upstream: nightly `dee37d89`, no hook, `--cpu-offload-gb 60`, `--kv-cache-dtype fp8_ds_mla` → **90.5 tok/s C1 (cache loaded; 91.3 live) · 302 C8 · ~420 C16 · KV 3.09M tokens at 1M**, 7.5-min bind on a cache hit. fp8 KV restores DSpark acceptance hook-off exactly as hook-on (tool_json 0.855 vs 0.810 nvfp4, shell 0.908 vs 0.829) at −4.6% on the prose knee; the live-vs-loaded tax is only −0.9% without the hook (4th pair), which points the FlashInfer cache-load tax at the hook's extra kernel shapes. Reference stays v18 (171 C1) — the hook is the lane; the flag-only row is what a Station owner gets from the yaml alone.

Also found: the hook's 10 GiB host guard clears by only 1–2 GiB on every 0909 boot and the nightly fell under it at rehome layer 9 — `--cpu-offload-gb 54` (9 UVA layers, identical post-rehome residency) is the workaround; offload GiB does not enter the kernel-shape hash. Three overnight-runner bugs are recorded in the bundle README (heredoc without `-i` silently skipped T3b; `e2c_heldout.py` called without its tag so T4's held-out never ran; `pgrep -f` matching its own shell). No window before this round recorded GPU power/cap/clock; from T3b5 on every runner does (Station power sloshing is a real mechanism — see limits).

## Release notes — Many Seat, v18 (2026-09-18)

v15 unchanged (hook, off60, util 0.97, 1M ctx, k-schedule shape) plus a scheduler that can hold a 24-seat workload: `--max-num-seqs 24`, `num_speculative_tokens_per_batch_size [[1,4,5],[5,24,1]]`, and `--cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128` (token counts). Trigger was the [35-seat fund workload post](https://al-engr.com/gb300-35-seat-fund-workload.html): this lane read warm agent-turn p95 8.5 s at 16 streams and p50 27 s at 24. **That was a slot queue at 16 seqs (KV usage peaked at 16 %), not prefill** — idle cold prefill is 22–23K tok/s from 26K to 207K tokens on every profile. Same-window on the fund harness: C24 warm p50 **27 → 0.55 s**, cold 120K under 16 streams p95 **19 → 9.2 s**, C16 warm p95 1.86 s, tools 64/64. Knee prose C1 **172** (v15 153, +12.6 %), C8 655 (v15 698, −6.2 %, unexplained), C16 945 (flat). KV **2,502,950 tokens** (2.4× at 1M) against 1.79M with default 24-seq graphs (v17) and ~2.26M on v15. Per-seat prose p10 at 16 streams is still ~16 tok/s: a long-context and tool lane, not a 16-seat decode lane. Four boots, one axis each; v16 (32 seqs + `long-prefill-token-threshold 2048`) and v17b (capture list in sequences, not tokens) failed and are kept as negatives. Bundle: [`results/2026-09-18-many-seat-v16-v17-v18/`](results/2026-09-18-many-seat-v16-v17-v18/).

## Release notes — Pin Hot Experts, v15 (2026-09-17)

v14 launch plus a bind-mounted hook that re-homes each MoE layer's experts by measured usage (295 hot rows in HBM, 89 cold rows pinned in Grace) and runs the MoE as two `do_finalize=False` routed-kernel calls plus one fp32-FMA finalize. Bit-identical to v14 (8/8 greedy prompts; acceptance rates identical on prose/structured/code/shell). **Confirmed** on two same-window pairs (2026-09-17): C1 88.7→153.3 (+72.7%, four runs within 0.4), C8 305→705, C16 386–417→~955 (six of seven reps; one 667). First pair (E2b): C1 89.3→153.1 (+71%), C8 308→701, C16 414→809, prose fixture 91→153. KV 4.75 GiB, pinned 62.3 GiB, 21 new autotune profiles (12 min). Design note: [vllm-pin-hot-experts](https://al-engr.com/vllm-pin-hot-experts.html). Confirm bundle: [`results/2026-09-17-e2c-v15-confirm/`](results/2026-09-17-e2c-v15-confirm/). First-pair bundle: [`results/2026-09-17-e2b-pin-hot-experts-v15/`](results/2026-09-17-e2b-pin-hot-experts-v15/).

**Confirmed (E2c):** second same-window pair; 50-prompt parity (44/44 non-tool exact on every pair; tool-prompt jitter present v15-vs-v15 too); round-7 gauntlet 36/36; held-out math +41.4%, JP/DE −1.55% (wash). **Corrections:** C16 moved ~2×, contrary to the "C1–C4 lever" expectation; autotune was 12 min, not 150.

## Round 7 — three matched boot pairs (September 15, 2026)

v14 averaged **231.7 vs 196.2 tok/s** on initial recorded-history replay and **260.4 vs 217.8 tok/s** on identical-request repeats: mean paired gains **18.1% / 19.5%**, consistent in all three pairs. Both profiles passed **36/36 attempts on twelve unique closed-loop tasks**; mean suite wall time was **11.78 vs 14.79 seconds**. v13 retained lower median replay TTFT. All 3,600 replay captures and 72 task attempts were audited; both references are preserved and stopped. These are bounded workload results, not production capacity or broad quality equivalence. [Full findings and public-safe evidence](results/2026-09-15-round7-matched-results/README.md).

## Release notes — Sixty-K Agent, v14 (2026-09-15)

One number changed: `KSCHED='[[1,4,5],[5,16,1]]'` — keep k=5 through four concurrent sequences, k=1 from five. Round 4 found v13 losing 9–12% on real Hermes agent transcripts at four workers (agent text accepts ~85% of drafts, so k=1 caps the step at one token); Round 5 confirmed it on a second pass and measured the fix. v14 on the same replay: 207 tok/s mean (v13 188, v12 207). C6–C16 knee identical to v13 (272 / 318 / 379 / 434). What it gives back: the C3/C4 prose gain (155 / 176, i.e. v12 level, vs v13's 195 / 225). Same autotune hash as v12/v13 — 454 s boot. Pick by traffic: agents at three-to-four streams → v14 (this lane's reference); batch prose throughput → v13. Bundle: `results/2026-09-15-round5-v14-agent-schedule/`.

**Round 6 — corrected interpretation (September 15, 2026):** the shuffled recorded-history proxy measured v14 171.1/174.1 tok/s versus v13 150.0/151.7 (+14.4% on that workload). An ordered proxy measured v14 **232.6 tok/s on its first measured pass and 275.2 when repeating the same requests**, with cache-hit ratios 71.3%/94.3% and TTFT p90 0.77/0.47 s. These are not live-agent capacity figures: tool history is flattened, tools are not executed, and generated answers are not carried forward. Tool mix and context also changed between fixtures, so the gain is not an isolated cache effect and the v13/v14 ranking was not tested on ordered sessions. The original bootstrap targets a busy-worker estimate, not wall-clock uncertainty. **The exact offload floor is unproven:** 55 failed the 1M minimum-KV check (2.2 GiB required, 0.65 available); 60 works; intermediate settings and shorter contexts remain open. These statements replace the earlier capacity/cache-causality/floor claims. Raw receipts are unchanged; see [`results/2026-09-15-round6-instrument-v2/`](results/2026-09-15-round6-instrument-v2/README.md).

## Release notes — Sixty-K (2026-09-14)

**v13 is v12 plus one line:** `num_speculative_tokens_per_batch_size=[[1,2,5],[3,16,1]]` — DSpark drafts 5 tokens while one or two sequences are running and 1 token at three or more. Same offload, same KV, same weights and verifier; this schedule happens to share v12's autotune hash (other breakpoints retune — see limits). Two same-window pairs against v12: C1/C2 unchanged, **C4 +27%, C8 +26%, C12 +27%, C16 +35% (429 tok/s)**. Fixture acceptance identical (the tool_json −7% seen in Round 3 was fixture ordering — identical on v12 and v13 in every position, Round 4). **Caveat (Round 4, same night):** on real Hermes agent transcripts at 4 concurrent workers v13 measures **−12% vs v12** (207 vs 236 tok/s) — agent text accepts ~85% of drafts, so k=1 leaves tokens on the table that k=5 collects even at C4. Prose/mixed lanes: v13. Agent-heavy lanes at 3–4 streams: v12 flags, or `[[1,4,5],[5,16,1]]` (untested). Two alternate breakpoint sets both lost on the knee (C3 −9% / −20%); v13's are right for prose. Details: [`results/2026-09-14-round4-v13-validation/`](results/2026-09-14-round4-v13-validation/README.md). Also in this round: `--async-scheduling` is a wash (±3%), and a cache-independent depth map shows **decode flat from 6K to 425K tokens (−7%)** — attention is not the lever on this box, expert fetch is. Details: [`results/2026-09-14-round3-ksched-depth/`](results/2026-09-14-round3-ksched-depth/README.md).

## Release notes — Sixty (2026-09-12)

This was the configuration to run before v13: `OFFGB=60 UTIL=0.97`, DSpark k=5, 1,048,576 context, Engram in Grace. v13 keeps all of it and adds the k-schedule.

| axis | verdict | evidence |
|---|---|---|
| offloaded-expert bytes (the lever) | ~~60 GiB is the floor at 1M~~ — corrected: 60 is the lowest verified setting here, not an exact floor; −12.7 GiB over C2C = +14.7% C1 in a same-window pair; 40 does not fit even at 131K | [`night-two-ledger.md`](results/2026-09-12-v12-1M-k5-off60-util97/night-two-ledger.md) |
| expert residency (hot in HBM, cold in Grace) | **closed** — ATS at 340 GB/s has no placement knob; every GPU-page-table path is ~90 GB/s; managed memory thrashes to 155 GB/s oversubscribed | [`spikes/`](results/2026-09-12-v12-1M-k5-off60-util97/spikes/) |
| speculation depth | **closed at k=5** — k=1 wins prose and multi-stream by 23–33%, k=5 wins agent text by a third; adaptive verification works but its forced 7 GiB graph capture costs more residency than it earns | [`ksweep/`](results/2026-09-12-v12-1M-k5-off60-util97/ksweep/), [vllm#56626](https://github.com/vllm-project/vllm/issues/56626) |
| where the step goes | MoE expert streaming 64% of GPU time at C1, 88% at C8; Grace fetch ≈ 9 ms of a 24 ms step | [`profile/`](results/2026-09-12-v12-1M-k5-off60-util97/profile/), [`routing/`](results/2026-09-12-v12-1M-k5-off60-util97/routing/) |
| host-side (THP, unpinned, rust frontend, language-model-only) | nothing adopted | [`failure-ledger.md`](research/failure-ledger.md) |

Not scheduled: a cuDNN discrete-mode MoE backend (per-expert pointers; the only way to cash the routing skew; 1–2 weeks), `--async-scheduling` (unmeasured; zero memory; likely small at C1), KV-dtype audit (capacity, not speed). Not pursued: REAP pruning, EGM/two-tensor row-map. The **batch-size K-schedule** became v13 on 2026-09-14 (+26–35% at C4–C16, C1 flat). `--async-scheduling` measured 2026-09-14: wash. The rebase probe onto `dsv41-optimized` is deprioritized by the depth map (decode flat to 425K; attention megakernels would not move throughput here).

Operating it: `docker update --restart unless-stopped dsv41-vllm-v13-1M-ksched-BOUND-REF`; health is `GET /v1/models` on the lane port; hot restart ~4 min, cold ~8 with the seeded autotune cache. Previous bound configs stay as stopped containers for rollback.

## What this runs

[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10` (since 2026-10-17; `df42c109` before), **as shipped** — 510 GB of MXFP4 routed experts, FP8 Engram table, FP8 attention. Nothing is re-quantized. It does not fit the ~250 GiB of HBM the GB300 exposes, so this recipe puts 60 GiB of expert weights (v12; v11 used 70) and the 189 GiB Engram table in Grace LPDDR5X and lets the GPU read them over NVLink-C2C through vLLM's UVA offload backend. Full native context (1,048,576 tokens) stays on. The in-checkpoint DSpark drafter runs at k=5.

The result is a single-box agent lane: prose decode is C2C-bound at ~89 tok/s, but the text agents actually emit — shell, code, tool-call JSON — runs 140–160 tok/s because the drafter accepts 64–91% of its guesses there. The number to quote is the one for your workload.

## Hardware

Profile: [`hardware/dgx-station-gb300.yaml`](../../../hardware/dgx-station-gb300.yaml) · snapshot: [`results/2026-09-10-v11-1M-k5-lpt6144/system.json`](results/2026-09-10-v11-1M-k5-lpt6144/system.json)

| | observed |
|---|---|
| GPU | NVIDIA GB300, **250.7 GiB visible** (nvidia-smi 256,703 MiB; spec says 288) |
| Host | Grace 72× Neoverse-V2, 494.5 GiB LPDDR5X, NVLink-C2C |
| OS / kernel / driver | Ubuntu 24.04.4 · 6.17.0-1032-nvidia-64k · 595.84 · CUDA 13.0 in-container |
| Storage | **checkpoint on local NVMe** (RAID0 2× SN850X, XFS). From a CIFS share the load is 10× slower and the boot is 92 min. |
| Do not | set `CUDA_VISIBLE_DEVICES` — DGX OS already hides the display GPU |

## Software

| | pin |
|---|---|
| Image | `vllm/vllm-openai:deepseekv41-flash-0909` @ `sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d` (day-0 build; serving v21 → v22 → **v23**). PRIOR documented release v20: `nightly-2671fedf…` @ `sha256:91c87506…`. Candidate, not this pin: `nightly-39983a5e` @ `sha256:a3bc7f09…` (config C, 2026-10-18) |
| vLLM | `0.1.dev20904+g179dd0fa9` (v20 was `0.29.1rc1.dev9+g2671fedfc`) |
| Libraries | torch 2.13.0+cu130 · transformers 5.17.0 · flashinfer 0.6.18 · triton 3.7.1 · env `PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024` (v23) |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, 253 files, 510,311,114,235 bytes (stat sum 2026-10-18); previous `df42c109…`, 510,313,343,553 bytes, byte-verified |

### Upstream watch (2026-09-20 evening) — one item bears on the parked v19 finding

Full table: [`research/upstream-watch-2026-09-20.md`](research/upstream-watch-2026-09-20.md). Pin is `nightly-dee37d89` (main 2026-09-18 04:32Z); newest nightly is `d05da62e` (2026-09-20 06:00Z). **vLLM #49435 "Fix SM100 fp8_ds_mla cache scales" merged 2026-09-20 00:15Z, after our pin**: the generic `concat_and_cache_ds_mla` writer stored arbitrary fp32 tile scales that the SM100 reader rounds to E8M0, so writer and reader could disagree on the quantization scale. v19 is the first lane here with `--kv-cache-dtype fp8_ds_mla`; DSV4.1's own compressed-cache Triton writer already used power-of-two scales, so whether any v19 KV group hit the affected kernel is unverified — but it is the cheapest physical candidate for the [tool-shaped logit drift](results/2026-09-20-fidelity/OPEN-FINDINGS.md) and costs one boot (`d05da62e` + hook + off54 + fp8 KV, new autotune hash, ~75-min live tune) plus the per-class `tf_logprob` split. Also post-pin: #56227 (SWA bounded replay — output after a prefix-cache hit is not bit-identical to cold *by design*; parity gates on rebased images must be cold-vs-cold), #57604 (MegaMoE prefill staging), #56625 (ViT CUDA graphs; PR says `FLASHMLA_MEGA_ATTN_DSV41` is broken upstream for this checkpoint — name the attention backend from the boot log on every rebase), #43310 (`--per-request-spec-decode-metrics`, opt-in — the right instrument for the per-request tool_json/shell acceptance bar). #37190 still open and not our path; FlashInfer autotune v2 is v0.7.0-only, so the pinned autotune set stays the workaround. HF model main is `dba1be0a` (encoding-only: tool namespaces) — weights unchanged, pin stands. Upstream's own recipe page (updated 2026-09-20) now defaults Blackwell to `--kv-cache-dtype fp8`, i.e. the v19 KV choice.

### Image pin is a moving target — rebase candidate identified (2026-09-13)

This image was built from vLLM staging branch `dsv41-feat` @ `e47aa780`, which no longer exists
under that name: it was replaced by **`dsv41-optimized`** (HEAD is the same `e47aa780`,
2026-09-10). DeepSeek-V4.1-Flash reached **vLLM main** the same day (#56228, then **#56214
"[Model] Support DeepSeek-V4.1-Flash" merged 2026-09-11**), so main is now *ahead* of this pin
on model support while the remaining perf kernels stay staged on `dsv41-optimized`.

Not yet tested here. The candidate is `vllm/vllm-openai:nightly-2671fedf…` (2026-09-13 06:15Z;
cu129 sibling 06:27Z), which carries #56228/#56214 plus DeepSelect top-k (#56464) and the Triton
input-metadata fusion (#56562). It predates #56512 (Engram prefetch/DP-shard) and #56682, so
those want the next nightly. **v0.29.0 will not work — it predates the DSV4.1 merge.** All flags
in this recipe still map to current mainline names; `--engram-config '{"cpu_offload":true}'`
becomes redundant once #56512 is in the image (CPU offload becomes the default there).

Everything above is measured against the pinned image and stands until a rebase is run
same-window against the live `-BOUND-REF`.

Full lock: [`results/…/software-lock.txt`](results/2026-09-10-v11-1M-k5-lpt6144/software-lock.txt) · exact command: [`launch-command.txt`](results/2026-09-10-v11-1M-k5-lpt6144/launch-command.txt).

## Launch

**v28-prob, this recipe's reference (since 2026-10-26).** The v26 launch below plus one key in
`--speculative-config`. `launch_br3.sh` is `launch_ar6.sh` with a `SPEC_EXTRA` splice
([`DIFF-br3-launch.txt`](results/2026-10-26-br4-probabilistic-drafting/scripts/DIFF-br3-launch.txt)); with `SPEC_EXTRA` empty it renders the v26 args
byte-for-byte. Same hook, seed map, GO/ctrl steps and autotune hash (hash-hit boot ~8 min).

```bash
NAME=dsv41-vllm-1017-v28-prob-BOUND-REF RUN=$HOME/dsv41/ar/v28 ROWMAP=rowmap-mixed-ar-v1.json \
SPEC_EXTRA=',"draft_sample_method":"probabilistic"' \
  bash results/2026-10-26-br4-probabilistic-drafting/scripts/launch_br3.sh      # boots FROZEN; then GO + ctrl.json exactly as for v26 below
touch $RUN/GO
echo '{"freeze": false, "baseline": true}' > $RUN/ctrl.json.tmp && mv $RUN/ctrl.json.tmp $RUN/ctrl.json
bash scripts/smoke_vllm.sh
```

- **Boot facts to check:** `GPU KV cache size: 2,481,694 tokens` (v26: 2,495,694; the −0.56% is the draft-logits
  cache), `Loaded 189 configs`, `draft_sample_method` present in `docker inspect` Args.
- **Speedup needs temperature > 0.** At temperature 0 the drafter's sample is its argmax and v28 behaves as v26.

**v26, the previous launch (2026-10-24; v28's rollback).** This is the v23 `docker run` with the hook directory and its env
changed. Edit the model, cache and hook paths at the top. The hook directory must contain the v26ar.2 hook
(sha1s in [`hook.txt`](results/2026-10-24-v26-seeded-adaptive/hook.txt)) and the seed map
[`rowmap-mixed-ar-v1.json`](results/2026-10-24-v26-seeded-adaptive/hook/rowmap-mixed-ar-v1.json).

```bash
NAME=dsv41-vllm-1017-v26seed-BOUND-REF RUN=$HOME/dsv41/ar/v26 ROWMAP=rowmap-mixed-ar-v1.json \
  bash results/2026-10-24-v26-seeded-adaptive/scripts/launch_ar6.sh      # boots FROZEN (counting only); hash-hit boot ~8 min
touch $RUN/GO
echo '{"freeze": false, "baseline": true}' > $RUN/ctrl.json.tmp && mv $RUN/ctrl.json.tmp $RUN/ctrl.json   # adaptation on, shipped policy
bash scripts/smoke_vllm.sh
```

- **Freeze without a reboot:** write `{"freeze": true}` to `ctrl.json`, or `{"restore_map": true}` to return to
  the boot map.
- **What changed vs the published v26ar.1 launcher:**
  [`launch_ar6-vs-launch_ar.diff`](results/2026-10-24-v26-seeded-adaptive/launch_ar6-vs-launch_ar.diff).

v23, the previous launch (one `docker run`; edit the model/cache/hook paths at the top):

```bash
bash results/2026-10-18-v23-pinexact-host/scripts/launch_v23.sh dsv41-vllm-1017-v23-pinexact-BOUND-REF
# = v22 + -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 ; hash-hit boot ~5 min
bash scripts/smoke_vllm.sh
```

The older env-driven form (v18 shown; v22/v23 change `KSCHED` to `[[1,8,5],[9,24,1]]` and add the env var):

```bash
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40 \
TAG=v18-many-seat OFFGB=60 UTIL=0.97 SEQS=24 SPEC=dspark:5 KSCHED='[[1,4,5],[5,24,1]]' CTX=1048576 \
EXTRA='--long-prefill-token-threshold 6144 --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128' \
bash scripts/launch-dsv41-vllm.sh
# plus the v15 pin-hot-experts hook: bind-mount sitecustomize.py + hook dir, PIN_MODE=split PIN_ROWMAP=/w/rowmap-static-v1.json
# (results/2026-09-18-many-seat-v16-v17-v18/launch-many-seat.sh is the exact launcher). ~6 min hot restart; a seqs change is a new autotune hash, ~16 min. Then:
bash scripts/smoke_vllm.sh
```

The server command inside the container (v23: `PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024` in the env, schedule `[[1,8,5],[9,24,1]]`; v18 shown):

```
vllm serve /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,24,1]]}' \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 --port 30006
```

Flags that matter, and why:

- **`PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024`** (v23). Pins allocations above 1 GiB at exact size instead of the next power of two: the two 94.4 GiB Engram tables stop costing 128 GiB each. −75 GiB pinned host, +~186 GiB at the hook's boot guard, zero GPU-side change.
- **`--max-num-seqs 24`** (v18). At 16 this lane queued on a 35-seat workload — warm agent-turn p50 27 s at 24 streams — while KV usage peaked at 16%. 24 slots: 0.55 s. Do not fix the same symptom with a lower `--long-prefill-token-threshold`: 2048 cost idle prefill −47% and prose p10 −29% (v16).
- **`--cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128`** (v18). Token counts, not sequence counts. Default graphs for 24 slots cost 4.1 GiB and KV fell to 1.79M; this list costs 2.4 GiB and KV is 2.50M with decode unchanged. A list that tops out at 24 (v17b) leaves every batch above 12 seqs uncaptured.

- **`--cpu-offload-gb 60` with `--gpu-memory-utilization 0.97`** is v12's notch. ~~60 → −2.61 GiB at 1M+DSpark~~ — that was measured at util 0.94. At 0.97 it fits: KV 4.89 GiB = 2.33M tokens = 2.2 concurrent 1M requests, and the 12.7 GiB of experts that move back into HBM buy +12% single-stream decode in a same-window comparison (89.2 vs 79.5). 40 still does not fit even at 131K. The fetch tax is ~4 ms of an ~11 ms step (~36%), measured against an all-HBM two-Station run; util alone moves nothing (at 70 it only grew KV). If you need more than two concurrent 1M contexts, use the v11 flags (`OFFGB=70 UTIL=0.94`, 82 tok/s, 4.5× at 1M).
- **`--max-model-len 1048576`** costs 5.5 GiB of KV pool relative to 131K and nothing in decode speed. There is no reason to run this model small.
- **`--speculative-config dspark k=5` + batch-size schedule `[[1,2,5],[3,16,1]]` (v13).** The drafter is in the checkpoint (`mtp.*`). At one or two streams k=5 wins agent text by a third over k=1 and loses prose by 8%; at three or more streams k=1 wins every point by 23–33% because a k=5 verify window multiplies unique experts fetched per step 3.7× and the batch already fills the step. The schedule takes each where it wins: C1 unchanged, C4–C16 +26–35%. If your lane is single-stream prose only, static k=3 (~+10% prose) is still a fair choice.
- **`--long-prefill-token-threshold 6144`** is what makes one lane serve mixed traffic. Without it a 480K prefill starves a short request to 18 s TTFT (vLLM [#51454](https://github.com/vllm-project/vllm/issues/51454)); with it, 0.9–1.1 s, and the long prompt pays +16%. The older `--max-num-partial-prefills` flags are gone from this vLLM.
- **`--tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice`** — without these, agent turns come back as narrated text.

### The autotune trap

FlashInfer autotunes the MXFP4 MoE kernels and caches the result under a hash of the engine config. **Every** change to `--max-num-seqs`, the speculative config, or `--max-model-len` produces a new hash, an empty directory, and a **74-minute** retune (py-spy: `trtllm_fp4_block_scale_moe` at 39 ms per candidate × thousands). Mount `/root/.cache/vllm` persistently, and when you change config, copy `autotune_configs.json` from a previous hash dir into the new one **before** the tuner reads it — the window is roughly 60 s after `Graph capturing finished`. The log receipt you want is `Loaded 210 configs`; if you see `Autotuning process starts` and then nothing for a minute, you missed and should kill and relaunch. We missed it four times in one day. This recipe's hash is `08c89d94`.

## Verify

| gate | how | last pass |
|---|---|---|
| schema | `scripts/check_recipe.py` | 2026-10-25 |
| digest | image `sha256:00d577a6…` (0909) by `docker image inspect`; `/model` is the 1017 revision `1bae0e10`; hook `b5e4f4f8…` read inside the container | 2026-10-18 |
| health | `/v1/models` 200; log shows `Available KV cache memory: 10.05 GiB`, `Loaded 210 configs`; `smoke_vllm.sh` 5/5 | 2026-09-11 |
| quality | smoke (arith 323, count 1–60, prose, parsed `tool_call`, thinking→`36`); **Hermes harness 10/10** tool calls with correct answers and side effects ([harness-summary.md](results/2026-09-12-v12-1M-k5-off60-util97/harness-summary.md) (v12 re-run 2026-09-12, 10/10)). No requant, so no divergence gate. | 2026-09-11 |
| performance | v23: bracketed CTL → CAND → CTL, RT C1/C4/C16 and replay r2 within −2%, knee6 C8/C6 on ≥ 4 reps within −2% (Card S0/S0c) | 2026-10-18 |
| fidelity (v23) | greedy non-tool parity 16/16 vs both controls; gauntlet 12/12; BFCL dev 92.33 / held-out 77.65; GSM8K-200 97.0 | 2026-10-18 |
| health (v26) | boot log: `Loaded 189 configs` (hash 62426808), `GPU KV cache size: 2,495,694 tokens`, `PIN_HOT installed PIN_MODE=adaptive … rowmap=/w/rowmap-mixed-ar-v1.json`, rehome 206.61 / 62.33 GiB; `steps.bin` grows after `GO`; serving-log error count 0 | 2026-10-24 |
| performance (v26) | Card AR6 v2, registered bars, same window: H1 held-out mixed C1 ≥ ×1.2 (×1.445), agent within 2 % (−0.7 %), adaptive steady state within 2 % of the static boot (−0.2 / −0.1 %) | 2026-10-24 |
| fidelity (v26) | gauntlet 12/12 on the fixed seed map and 12/12 live under adaptation; greedy parity 49/50 static vs seed (documented noise prompt); 0 stale swaps, 0 server errors. BFCL / GSM8K / TF not re-run | 2026-10-24 |
| soak (v26) | **waived** at promotion (James); longest unattended window ~1 h | — |
| performance (v28-prob) | Card BR4, registered bars, bracket A → P → PB → A2: per-class temp-1.0 decode C1/C8 all 10 cells ≥ ctrl − bar (P 1.045–1.187×), mixed C1 +0.68% (bar −2%), tokens/step 1.103× | 2026-10-26 |
| fidelity (v28-prob) | TF top-1 flips 1.10–1.13% vs floor 1.136%, margin flips 0, gauntlet 12/12, replay turns [6, 8] in range 6–8, temp-0 parity on every class, 0 errors; BFCL / GSM8K / soak not re-run | 2026-10-26 |

### Promotion gates (amended 2026-10-18, accepted by James)

- **Per-class DSpark acceptance is an investigation trigger, not a gate.** Any class moving more than 2 pts — on the
  8-prompt agent fixture or on the per-class instrument's `/metrics` acceptance — requires a written explanation before
  promotion. It does not block by itself. (Why: the 8-prompt fixture flagged code −4.7 pts on every nightly variant; 40
  real prompts per class did not reproduce it.)
- **The gate for drafter-side changes is per-class decode wall-clock**: `perclass_run.sh` with the frozen prompt set
  (sha1 `618898aa`, real Hermes transcript turns, private), forced length 256, C1 and C8, r1/r2. For every class × cell,
  candidate mean ≥ control mean − max(2%, 1.5 × the control's same-config spread); same-window v23 control, bracketed
  when a boot is involved; all 10 cells must pass. Instrument: [`results/2026-10-18-p2-nightly-39983a5e-gate/scripts/perclass/`](results/2026-10-18-p2-nightly-39983a5e-gate/scripts/perclass/).
- **Fidelity gates unchanged:** BFCL ≥ control − 1 pt (dev and the frozen held-out sibling), GSM8K, margin-weighted
  teacher-forced flips, gauntlet 12/12. Speed bars outside the drafter question (RT harness, replay, knee) unchanged.
- **C1 headlines** are a mean of 3 with acceptance alongside and a per-class split (`docs/benchmarking.md`).

**On instruments.** `knee.sh` runs each concurrency twice and the two runs agree within ~1%; it is the decision metric. `replay.py` replays 24 real agent turns and is useful as a workload-shaped smoke test, but at temperature 0 it swings **±17 tok/s** run to run (batched MoE + speculative decode are not bit-deterministic). We published an overnight "win" on it and retracted it the next morning when the knee said the opposite. Bench on the tight instrument.

## Results

### v28 — Probabilistic drafting (this recipe's release since 2026-10-26; not served on `:30006`)

The launch is v26 + one key: `launch_br3.sh` with `SPEC_EXTRA=',"draft_sample_method":"probabilistic"'` (see Launch).
Container `dsv41-vllm-1017-v28-prob-BOUND-REF`, stopped-and-kept; rollback v26seed. Bundle:
[`results/2026-10-26-br4-probabilistic-drafting/`](results/2026-10-26-br4-probabilistic-drafting/README.md); every cell is in [`throughput.csv`](results/2026-10-26-br4-probabilistic-drafting/throughput.csv).

The instrument is the per-class temperature-1.0 window (40 private Hermes-transcript prompts per class, 256 forced
tokens, seeded) and `ar_mixed_t.py` at temperature 1.0, against bracketing v26 controls in the same window.

| cell | v28-prob | v26 controls A / A2 |
|---|---|---|
| accepted tokens per verify step, temp 1.0 | **3.438** | 3.122 / 3.115 |
| tool_json decode C1 / C8, temp 1.0 | **294.0 / 791.8** | 254.6 / 251.4 · 671.9 / 661.9 |
| prose decode C1 / C8, temp 1.0 | **217.2 / 494.3** | 197.4 / 197.2 · 451.8 / 454.6 |
| shell_ops decode C1 / C8, temp 1.0 | **319.8 / 723.3** | 291.9 / 282.6 · 642.2 / 635.8 |
| mixed C1 tok/s, temp 1.0 | 181.09 | 179.79 / 179.93 |
| acceptance at draft position 1 / 5 | **.785 / .278** | .699 / .245 |
| TF top-1 flips vs controls | 1.129% / 1.098% | floor 1.136% |
| KV at 1M, boot | 2,481,694 tokens (2.37×) | 2,495,694 (2.38×) |

Temperature 0 is unchanged from v26 (G2b parity PASS on every class). Not re-run on v28: BFCL, GSM8K, long context,
C16+, a soak.

### v26 — Seeded Adaptive (previous release, 2026-10-24 → 2026-10-26; v28's rollback; served `:30006` 2026-10-24 ~21:30–23:16 CDT)

2026-10-26: block verification on v26 (Card BR3) is a null, with 0.999× tokens per verify step at temperature 1.0. It was
not adopted; see the 2026-10-26 round and [`results/2026-10-26-br3-block-verification-null/`](results/2026-10-26-br3-block-verification-null/README.md).

Bundle: [`results/2026-10-24-v26-seeded-adaptive/`](results/2026-10-24-v26-seeded-adaptive/README.md); every cell is in [`throughput.csv`](results/2026-10-24-v26-seeded-adaptive/throughput.csv).

The instrument is the AR shift/mixed harness: agent-English prose and JP/DE essay pools, synthetic. It is not the
kbench CNN/DM real-text cell of the v23 table below, so compare within one table. kbench, knee and BFCL were not
re-run on v26.

| cell | v26 (seed map) | static-map control, same window |
|---|---|---|
| held-out mixed C1 aggregate, maps frozen | **183.73** | 127.12 |
| held-out agent / JP/DE per-request median, frozen | 201.36 / 177.23 | 202.74 / 85.45 |
| first JP/DE request after boot | 177.89 | 84.96 |
| adaptive steady-state C1, agent / JP/DE | **218.94 / 189.59** | 219.13 / 189.94 |
| cold-share T90 agent → JP/DE / JP/DE → agent | 8.7 / 14.0 s | 11.0 / 14.0 s |
| peak cold share after the shift | 5.7 % / 3.3 % | 25.4 % / 6.0 % |
| swap churn (shipped policy) | 95 /min | 107 /min |
| swap-bearing host steps / p99 host step (whole boot) | 2.53 % / 22.17 ms | 2.50 % / 24.42 ms |
| KV at 1M, boot | 2,495,694 tokens (2.38×) | same |

### v23 — Pin Exact (serving 2026-10-18 → 2026-10-22; the base v26 launches from)

Bundle: [`results/2026-10-18-v23-pinexact-host/`](results/2026-10-18-v23-pinexact-host/README.md). Real-text harness = RonanLabs `kbench_chat_ds.py` (CNN/DM ~8.2K in, 1,024 forced, thinking off, T=0, nonce, 0% prefix hits). Not slot-capped below C24.

| cell | v23 | window |
|---|---|---|
| RT C1, mean of 3 (accept/step) | **188.2 (2.99)** | S0c promotion boot; later controls 186.1 (2.91) live, 186.8 (2.94) restart |
| RT C4 / C16 / C32 output | 325.1 / 523.5 / 506.5 | Card S0 |
| TTFT 8K (RT C1 median) | 0.643 / 0.647 s | Card P2f A / D |
| knee6 C8, 4 reps | 596.5 / 614.0 (bimodal reps) | Card P2f A / D |
| knee r2 C1 / C8 / C16 (prose) | 192.2 / 618.5 / 944.4 | Card P2f A |
| cold ~53K-token prefill | 24.1K tok/s | Card P2f A |
| replay r2, 4 workers | 438 | Card S0 |
| pinned host / KV tokens | 252.79 GiB / 2,504,891 | Card S0 |

### v18 — Many Seat (superseded; v22 is the rollback since 2026-10-18)

Same-window idle knee re-check 2026-09-18 (v17 ×2 then v18 ×2, box otherwise idle) with a same-afternoon v15 control. Knee prompt class **prose**, T=0, 192 tokens. Not slot-capped at C16 (`--max-num-seqs 24`). Bundle: [`results/2026-09-18-many-seat-v16-v17-v18/`](results/2026-09-18-many-seat-v16-v17-v18/).

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| v18 | 171.8 / 172.1 | 653.8 / 656.9 | 747.5 / 944.6 |
| v17 | 164.9 / 165.1 | 640.6 / 658.7 | 878.1 / 949.7 |
| v15 (control) | 152.7 / 152.9 | 696.8 / 699.3 | 942.9 / 943.7 |

Fund harness (35-seat mix on real 10-K text; means of two v18 windows around a v17 control, v15 control trailing): C24 warm agent-turn p50 **0.55 s** (v17 0.78, v15 27.3); C16 warm p95 1.86 s (v15 1.88; morning v15 run 8.54); cold 120K under 16 streams p95 9.2 s (v15 19.1); C16 prose p10 15.75 (v15 16.5); tools 64/64 everywhere; KV 2,502,950 tokens. Ladder and the two failed boots (v16, v17b) are in the bundle README.

### v15 — Pin Hot Experts (retired 2026-09-18; v18 = v15 + scheduler)

#### Confirmed (E2c)

Two same-window pairs 2026-09-17, v15a → v14 → v15b, knee ×2 each. Knee prompt class: **prose**, temperature 0, 192 tokens (`knee.sh`: "Write a detailed paragraph about the number N"). C16 is slot-capped by `--max-num-seqs 16`; `max_running_requests` was not set. Bundle: [`results/2026-09-17-e2c-v15-confirm/`](results/2026-09-17-e2c-v15-confirm/). Design: [vllm-pin-hot-experts](https://al-engr.com/vllm-pin-hot-experts.html).

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| v15a | 153.5 / 153.3 | 704.6 / 701.6 | 952.6 / 956.7 |
| v14  | 88.7 / 88.8 | 304.7 / 305.8 | 385.5 / 417.3 |
| v15b | 153.1 / 153.2 | 703.0 / 706.5 | 812.1 / 957.8 |

C1 two-window v15 mean **153.28** vs v14 **88.74** = **+72.7%** (four runs within 0.4). C16 four v15 means 952.6 / 956.7 / 812.1 / 957.8; the 812 is one mixed pair (internal 667.4 / 956.8). Other six internal C16 reps 952–958.

Held-out (streaming first→last token, 5×400 tok, not in the E2a profile): math/proof English **+41.4%** (129.78 → 183.5); non-English JP+DE essays **−1.55%** (93.02 → 91.58) — a wash vs the −1.5% bar (0.05 tok/s). Parity 50 greedy: non-tool **44/44** exact on every pair; tool-prompt jitter present v15-vs-v15 too. Gauntlet **36/36**.

#### First pair (E2b) — one pair, kept

Same-window pair 2026-09-17 14:51–14:58 CDT, **one pair**. Bundle: [`results/2026-09-17-e2b-pin-hot-experts-v15/`](results/2026-09-17-e2b-pin-hot-experts-v15/).

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| v14 control | 89.3 | 307.7 | 414.3 |
| v15 candidate | 153.1 | 701.4 | 809.0 |

C1 **+71.4%**. C16 candidate runs 666.8 / 951.2 — spread is real; both still ≫ control 413 / 415.

| | v14 control | v15 |
|---|--:|--:|
| agent prose tok/s | 98.3 | 169.0 |
| structured | 123.0 | 207.6 |
| code | 153.9 | 213.6 |
| shell_ops | 160.2 | 250.3 |
| tool_json | 180.4 | 182.1 |
| weighted accept | 59.9% | 59.6% |
| T5 prose fixture | 91.4 | 152.5 |
| replay n=4 r1 / r2 | 170.6 / 222.7 | 247.5 / 431.3 |

Parity: greedy 8/8 vs v14. Accept rates bit-identical on prose/structured/code/shell. KV 4.75 GiB; pinned 62.33 GiB; autotune 12 min (21 new).

### v14 — v13 with the k=5 band extended to C4: `[[1,4,5],[5,16,1]]` — v14 (retired 2026-09-17)

| | v14 | v13 | v12 |
|---|---|---|---|
| Real-agent replay, 4 workers (mean of two runs) | **207** (184 / 229) | 188 (171 / 206) | 207 (188 / 225) |
| accepted/step on replay | 2.6–2.7 | 0.9 | 2.6–2.7 |
| Knee C3 / C4 | 155 / 176 | **195 / 225** | 155 / 179 |
| Knee C6 / C8 / C12 / C16 | 272 / 318 / 379 / 434 | 274 / 321 / 377 / 434 | 219 / 261 / 279 / 321 |
| C1 knee / fixture tool_json | 91.0 / 158.7 | 91.5 / 158.7 | 91.1 / 158 |
| Autotune hash / boot | 9ac7b387 / 454 s | 9ac7b387 / 151 s | 9ac7b387 |

v14/v13 same window 2026-09-15 03:14–03:38 CDT; v12 knee column from Round 4 (2026-09-14, same night). The replay instrument's run-to-run spread is ~20%, so the table shows both runs; the means are what the decision rests on. Container `dsv41-vllm-v14-1M-ksched-agent-BOUND-REF`.

### v13 — v12 + `num_speculative_tokens_per_batch_size=[[1,2,5],[3,16,1]]` (prose/throughput reference)

Run [`2026-09-14-round3-ksched-depth`](results/2026-09-14-round3-ksched-depth/) · raw: [`throughput.csv`](results/2026-09-14-round3-ksched-depth/throughput.csv) · two same-window pairs against the v12 container, 18:02 and 18:27 CDT.

**Decode (knee, prose prompts; mean of both pairs, v12 control mean in parentheses)**

| C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|
| **90.7** (90.6) | 127.2 (127.2) | **223.4** (175.9) | **319.1** (252.6) | **339.3** (266.5) | **429.0** (318.8) |

C1 fixture classes and acceptance are the v12 numbers (the schedule is k=5 there). Decode vs prompt depth on the same boot: 124 / 117 / 114 / — / 122 tok/s at 6.5K / 53K / 106K / 212K / 425K — flat. Memory picture identical to v12 (hash 9ac7b387 hit, KV 4.87 GiB).

**Round 4 (same night, [`2026-09-14-round4-v13-validation`](results/2026-09-14-round4-v13-validation/README.md)):** C3 192.6 / C6 272.9 vs v12 155.2 / 219.1 (+24 / +25%). Real Hermes transcripts (`replay_c.py`): 1 worker v13 116.5 vs v12 111.3 (wash); **4 workers v13 207.0 vs v12 235.9 (−12%)** — accepted/step 0.87 at k=1 vs 2.70 at k=5, because agent text accepts 85%. Breakpoint alternatives `[[1,3,5],[4,16,1]]` and `[[1,2,5],[3,4,3],[5,16,1]]` both lose on the knee (C3 −20% / −9%). Quote v13 for prose/mixed load; for agent-heavy 3–4-stream lanes the v12 flags are still the better measured choice.

### v12 — `OFFGB=60 UTIL=0.97` (superseded by v13, same flags minus the schedule)

Run [`2026-09-12-v12-1M-k5-off60-util97`](results/2026-09-12-v12-1M-k5-off60-util97/) · raw: [`throughput.csv`](results/2026-09-12-v12-1M-k5-off60-util97/throughput.csv) · how it was chosen: [`night-two-ledger.md`](results/2026-09-12-v12-1M-k5-off60-util97/night-two-ledger.md) · warm, on-box, measured against a same-window v11 control boot (79.5 C1) because the reference drifts ~3% day to day.

**Decode (knee, prose prompts, DSpark k=5)**

| C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|
| **89.2** | 124.9 | 173.4 | 233.7 | 270.8 | **311.4** |

**Decode by content class (C1, `agent_fixture.sh`)** — every class moved +10–12% with acceptance unchanged, which is what a pure bandwidth change looks like:

| class | tok/s | DSpark accept | tok/step |
|---|---|---|---|
| shell_ops | 160.3 | 90.8% | 4.54 |
| code | 153.7 | 63.8% | 3.19 |
| tool_json | 141.9 | 87.8% | 4.39 |
| structured | 122.7 | 49.5% | 2.48 |
| prose | 97.7 | 29.8% | 1.49 |

Cost: KV 4.89 GiB = 2.33M tokens = **2.2 concurrent full-1M requests** (v11: 4.5). Host 432/494 GiB. First boot on the new flags pays the 74-minute autotune (new hash `9ac7b387`); after that ~8 min.

Prefill, mixed-traffic and harness numbers were re-measured on v12 on 2026-09-12 (prefill 207K in 11.4 s, starve shorts 0.8–1.1 s under a 480K prefill, Hermes harness 10/10) and did not move — they are not offload-bound. See [harness-summary.md](results/2026-09-12-v12-1M-k5-off60-util97/harness-summary.md).

**Where the decode step goes** (torch profiler, k=5): MoE expert streaming is 64% of GPU time at C1 and 88% at C8; attention 17%/6%. A k=5 verify window touches 3.7× the unique experts of one token (router property, ±3% across categories), a quarter of them in Grace — which is why k=5 is +100% on shell and −11% on prose vs k=0. Details: [profile/](results/2026-09-12-v12-1M-k5-off60-util97/profile/README.md), [routing/](results/2026-09-12-v12-1M-k5-off60-util97/routing/README.md). **Speculation depth (k sweep + adaptive, 2026-09-12 morning):** static k=1 wins prose (+8%) and every multi-stream point (+23–33% C1–C16) but loses the agent categories by ~a third; DSpark adaptive verification (V2 runner, confidence head) lands between them with no losing category but misses the promotion bar, partly because the V2 runner's graph capture costs ~6 GiB of residency. v12 static k=5 stays the reference. Table and raw JSON in [ksweep/](results/2026-09-12-v12-1M-k5-off60-util97/ksweep/README.md). **Follow-up (2026-09-14):** replacing the boot-profiled verify-cost table with a running median of real verify-step times (env-gated patch, two same-window pairs) confirmed the boot table under-prices verify at decode sizes by 11–40%, but the controller only gains C1 +3.3% / C8 +1.7% and loses C2/C4 (−6/−8%); tool-JSON +15% is the one consistent mover. Not adopted. The missing term is unique experts per step, not milliseconds per token. [2026-09-14-onlinevc-adaptive-off66/](results/2026-09-14-onlinevc-adaptive-off66/README.md).

**Why the offload lever stops here:** three spikes (VMM/EGM mixed backing, managed memory) show that on GB300 the only fast GPU→Grace read path is the pinned/ATS one the UVA offloader already uses (~340 GB/s); every path that allows per-row placement runs at ~90 GB/s, and oversubscribed managed memory settles at 155 GB/s. Details and scripts in [results/…/spikes/](results/2026-09-12-v12-1M-k5-off60-util97/spikes/README.md).

### v11 — `OFFGB=70 UTIL=0.94` (baseline; use if you need >2 concurrent 1M contexts)

Run [`2026-09-10-v11-1M-k5-lpt6144`](results/2026-09-10-v11-1M-k5-lpt6144/) · raw: [`throughput.csv`](results/2026-09-10-v11-1M-k5-lpt6144/throughput.csv) · warm, on-box.

**Decode (knee, prose prompts, DSpark k=5)**

| C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|
| 82.1 | 119.2 | 164.5 | 237.0 | 251.6 | 286.5 |

Same-night re-measurement of this config on 2026-09-11 gave 79.5 / 229.4 / 281.7 — the ~3% drift band.

**Decode by content class (C1, `agent_fixture.sh`)**

| class | tok/s | DSpark accept | tok/step |
|---|---|---|---|
| shell_ops | 150.0 | 90.8% | 4.54 |
| code | 145.6 | 63.8% | 3.19 |
| tool_json | 131 | 87.8% | 4.39 |
| structured | 115.8 | 49.5% | 2.48 |
| prose | 92.8 | 29.8% | 1.49 |

Real Hermes transcripts (24 turns): ~49% acceptance, real tool-call turns 138–171 tok/s.

**Prefill (cold KV, thinking off)**

| tokens | TTFT | tok/s |
|---|---|---|
| 6,538 | 0.37 s | 17,900 |
| 103,814 | 5.38 s | 19,300 |
| 207K | 11.3 s | 18,300 |
| 415K | 24.9 s | 16,700 |
| **972,435** | **85.0 s** | 11,400 |

Zero preemptions at 972K; HBM peaked at 243 GiB. Warm prefix on an 8.8K-token system prompt: 0.665 → 0.360 s.

**Mixed traffic (`starve.py`)**: four short requests fired during a 480K prefill see 0.89–1.09 s TTFT (18.2 / 13.2 / 8.2 / 3.2 s without `--long-prefill-token-threshold`).

**Boot**: 8 min cold from local NVMe with a seeded autotune cache; 4 min hot `docker start`. From CIFS with a retune: 92 min.

## Known limits

See `limits:` in [`recipe.yaml`](recipe.yaml).

2026-10-26 addition (Card BR5):
- **replay_turns G6 with 2 passes is too narrow on this lane (see BR5).** The temperature-0 replay is not
  deterministic here: per-pass tool-turn counts ran 6–9 of 12 across seven arms of BR4 and BR5, with no relation to
  the block key, and a range drawn from two controls × two passes tripped PB once in each direction. PB (v28-prob +
  block) is retired on this lane until Card BR6 re-runs the bracket with 4 passes per arm and a pooled-control bar.

2026-10-26 additions (Card BR4, v28-prob):
- **The +10% is a temperature > 0 result.** At temperature 0 probabilistic drafting returns the argmax and v28 runs
  exactly like v26. Clients that send temperature 0 get nothing from it.
- **KV is −0.56% (2,481,694 vs 2,495,694 tokens).** vLLM allocates a bf16 draft-logits cache (~31 MB at 24 seqs × k=5
  × 129,280 vocab) before KV profiling. Max concurrency at 1M tokens goes 2.38× → 2.37×.
- **PB (+ block verification) is a candidate, not the release.** It was +1.5% tokens per step over P at temperature
  1.0, but it failed two pre-registered temperature-0 parity gates: G2b shell_ops C1 +3.6% vs a 2.6% bar, and G6
  replay tool-turns 9/12 in one pass vs a control range of 6–8. It needs its own bracket against v28-prob.
  *(Done 2026-10-26, Card BR5: G2b passed; G6 failed the other way, [6, 7] vs 8–9 → PB retired pending BR6.)*
- **Not re-run on v28:** BFCL, GSM8K, long context (128K / 1M), C16+, and a soak. The park boot verified facts and
  two smoke requests only.

2026-10-26 addition (Card BR3):
- **Block verification is a no-op with greedy drafting. Measure spec-decode changes at temperature > 0.**
  `rejection_sample_method: block` on v26 (greedy DSpark drafts) moved tokens per verify step by 0.999× at
  temperature 1.0. At temperature 0 block verification is never applied. Any stock temperature-0 speed instrument
  therefore reads a verifier change as a null by construction. The sidecar lane drafts probabilistically, and there
  the same key is +2.0%. Card BR4 tests probabilistic drafting on this lane. *(Done 2026-10-26: probabilistic
  drafting +10.3% tokens per step → v28-prob; block on top +1.5%, candidate.)*

2026-10-24 additions (v26):
- **The 48 h soak was waived at promotion** (James, 2026-10-24). The AR5 soak ran ~19:57 → ~20:4x CDT and was cancelled. The longest unattended window on v26 so far is ~1 h plus the Card AR6 / gauntlet battery traffic. Multi-day memory drift and multi-hour swap churn under real traffic are unmeasured.
- **Convergence correction.** The 2026-10-23 card and Part 1 said 60–90 s. That figure was a windowing artifact; re-analysis gives cold share < 1 % in 9–16 s, and AR6 measures T90 at 8.7 / 14.0 s on the seed boot.
- **Swap cost.** Swap-bearing steps run ~1.9× the p95 of swap-free steps (G2 failed against its 1.25 bar on every cell, including the control). 2.53 % of steps carry swaps, and whole-boot p99 host step is 22.17 ms vs 19.52 ms swap-free.
- **Not run on v26:** long context (no 128K or 1M request; the 950K needle in the 2026-10-24 receipts is the sidecar lane's), BFCL, GSM8K, teacher-forced Δlogprob, and C4+ concurrency.
- **Image of record** is still `vllm/vllm-openai:deepseekv41-flash-0909`, a pullable tag. The hook is python-side, with no kernel change, and is pinned by sha1.
- **v26ar.2 hook source is not in this repo yet.** It is v26ar.1 plus env-gated telemetry. The seed map is published.

2026-10-18 additions: the per-class acceptance rule is amended (trigger, not gate); vLLM nightly-39983a5e config C passed its gates and is a candidate, not the runtime (GSM8K at its bar, +48% completion tokens); P1 prefill graphs failed on KV; prose-knee C8 is bimodal (gate on ≥ 4 reps); the 0909 image renders thinking-mode `reasoning_effort` low/high as 25/50 where the 1017 checkpoint's encoder uses 50/75. The short version: decode is C2C + kernel bound and the offloaded-expert byte count is the lever (~0.76 tok/s per GiB moved into HBM; v12 spends the last HBM headroom on it and pays in KV); quote the tok/s for your content class, not the best one; the autotune hash trap is real; `reasoning_effort: medium` is rejected by the template (use `low|high|xhigh|max`); ~110 GB of host shmem was unaccounted for (corrected in part on 2026-10-18: 75 GiB of pinned memory was power-of-two rounding, removed by v23; v23 leaves ~200 GiB MemAvailable).

## What didn't work

[`research/failure-ledger.md`](research/failure-ledger.md) — SGLang at 3.3 tok/s, four vLLM UVA failures before the first bind, the offload bracket, the autotune misses, a first overnight of decode experiments (THP, unpinned host memory, `--language-model-only`, Rust frontend) that produced one real lesson about instruments and zero adopted flags, and a second night (k-sweep, util-only, off60) that produced v12 — see [`night-two-ledger.md`](results/2026-09-12-v12-1M-k5-off60-util97/night-two-ledger.md). A third night (2026-09-18/19) closed the k=3 mid-band schedule and on-line adaptive placement and left the nightly-image cache-load question open — [`2026-09-19-overnight-ksched-nightly-adaptive/`](results/2026-09-19-overnight-ksched-nightly-adaptive/README.md).

## Rollback

**v28-prob → v26:** `docker stop dsv41-vllm-1017-v28-prob-BOUND-REF && docker start dsv41-vllm-1017-v26seed-BOUND-REF`
(hash hit, ~8 min; then GO + ctrl as in Launch). Or relaunch with `SPEC_EXTRA` empty. A client can also get v26
behaviour on v28 by sending temperature 0.

**v26 → v23 behaviour, no reboot:** write `{"freeze": true}`, then `{"restore_map": true}`, to the run dir's `ctrl.json`. The hook stops swapping and returns to the boot map.

**Exact v23 runtime:** `docker stop dsv41-vllm-1017-v26seed-BOUND-REF`, then `docker start` the kept v23 container (hash hit, ~5 min).

**The live lane** is the sidecar sibling's v27 pair (since 2026-10-26 04:08 CDT); its rollback is the v25m-kvtier pair plus its fail-stop watchdog; see the sibling recipe. *(Corrected 2026-10-26: this line named the v25m-MIXMAP pair.)*

v23 → v22: `docker stop dsv41-vllm-1017-v23-pinexact-BOUND-REF && docker start dsv41-vllm-1017-v22-ksched1to8-RETIRED-REF` (~5 min, autotune hash hit). Every bound config is kept as a stopped container `dsv41-vllm-<tag>`. `docker stop` the lane, `docker start` the previous one; ~4 min. Production surfaces on other ports are never part of this lane.

## Credits

2026-10-26 round (Card BR4, v28-prob):
- **Probabilistic drafting and block verification as a pair:** the config of
  [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR), which
  set both `draft_sample_method: probabilistic` and `rejection_sample_method: block`.
- **Block verification:** Sun et al., as implemented in vLLM's `rejection_sample_method: block`.
- **Runner, verdict, park and write-up:** Milo (Hermes). Promotion of P as the reference: James ("go").

2026-10-26 round (Card BR3):
- **Block verification:** Sun et al. (block verification, a distribution-preserving verifier for speculative
  decoding), as implemented in vLLM's `rejection_sample_method: block`.
- **The idea to try it on this checkpoint:** [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR).
- **Runner, verdict and write-up:** Milo (Hermes). The desk brief and runner were drafted by a claude-opus-5-5
  subagent and reviewed before launch.

2026-10-24 round (Card AR6 v2, v26):
- **Campaign, runner, verdict and write-up:** Milo (Hermes) — the Part 3 byline lists one anthropic/claude-fable-5 session for measurements and campaign.
- **Promotion decision and soak waiver:** James.
- **Routing traces** of this exact checkpoint, which made the desk simulation possible: [05yuki](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers).
- **Half-life evidence:** N3mezis's trace replays.
- **Prior art for traffic-fitted maps:** original-el8's `rowmap-mix-v1`.
- **The device-side planner idea:** [vLLM #56177](https://github.com/vllm-project/vllm/pull/56177) by 01554.

2026-10-18 round: briefs, bars and write-up by Milo (Hermes, claude-fable-5-1); worker runs by claude-opus-5-5 subagents from those briefs; original-el8 for the P1 (prefill graphs) and P5 (long-generation cell) ideas; the Opus 5.5 review behind the per-class rule amendment; Ronan (RonanLabsAI) for the real-text RT harness; vLLM upstream for nightly-39983a5e, #58132, #56893 and the managed autotune store.

Campaign design, briefs, win bars, and verdicts: **Milo** (James Meadlock's Handler agent, Hermes `milo` profile, session model anthropic/claude-fable-5.1) with James. Worker runs (E1–E5 windows, recipe PRs): Grok 4.6 via xAI and claude-fable-5.1 via Nous, from Milo's written briefs; every number here was read back from the worker's JSON. The pin-hot-experts idea: James, in conversation with Grok.

DeepSeek for the model and the in-checkpoint DSpark drafter. The vLLM team for the day-0 image, UVA offload backend, and DeepSeek-V4.1 support. Community data points that shaped decisions: 0xSero and Tech2Wild (4× RTX PRO 6000 and DGX Spark builds), Fraser Price (`dspark-vllm`), LMSYS (Engram huge-page finding). Blog write-up: [al-engr.com](https://al-engr.com/gb300-deepseek-flash-41-testing.html).
