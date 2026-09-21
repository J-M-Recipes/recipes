# DeepSeek-V4.1-Flash at 1M context on one DGX Station GB300

**Reference release: Many Seat, v18** (2026-09-18: v15 hook + `--max-num-seqs 24` + token-sized `--cudagraph-capture-sizes`; 172 tok/s C1 prose, KV 2.50M tokens, C24 warm agent turn 0.55 s p50; **BFCL v4 tool-call exact-match 93.3%**) · **v19 "Nightly" was promoted and reverted on 2026-09-20** (see Round 10) · previous **v15 Pin Hot Experts** (retired 2026-09-18) · **v14 Sixty-K Agent** (retired 2026-09-17) · Historical **v13: 90 tok/s single-stream prose · 140–160 tok/s on agent/code text · 429 agg tok/s at C16** (v12: 89 / 140–160 / 311 · v11: 82 / 130–150 / 287) · 972K-token prompt prefilled in 85 s · Hermes tool-calling 10/10

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

[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) @ `df42c109`, **as shipped** — 510 GB of MXFP4 routed experts, FP8 Engram table, FP8 attention. Nothing is re-quantized. It does not fit the ~250 GiB of HBM the GB300 exposes, so this recipe puts 60 GiB of expert weights (v12; v11 used 70) and the 189 GiB Engram table in Grace LPDDR5X and lets the GPU read them over NVLink-C2C through vLLM's UVA offload backend. Full native context (1,048,576 tokens) stays on. The in-checkpoint DSpark drafter runs at k=5.

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
| Image | `vllm/vllm-openai:deepseekv41-flash-0909` @ `sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d` (upstream day-0 build, 2026-09-10) |
| vLLM | `0.1.dev20904+g179dd0fa9` |
| Libraries | torch 2.13.0+cu130 · transformers 5.17.0 · flashinfer 0.6.18 · triton 3.7.1 |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` @ `df42c109f1defefcbfcedbe7d905718a12266e40`, 510,313,343,553 bytes, byte-verified |

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

```bash
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40 \
TAG=v18-many-seat OFFGB=60 UTIL=0.97 SEQS=24 SPEC=dspark:5 KSCHED='[[1,4,5],[5,24,1]]' CTX=1048576 \
EXTRA='--long-prefill-token-threshold 6144 --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128' \
bash scripts/launch-dsv41-vllm.sh
# plus the v15 pin-hot-experts hook: bind-mount sitecustomize.py + hook dir, PIN_MODE=split PIN_ROWMAP=/w/rowmap-static-v1.json
# (results/2026-09-18-many-seat-v16-v17-v18/launch-many-seat.sh is the exact launcher). ~6 min hot restart; a seqs change is a new autotune hash, ~16 min. Then:
bash scripts/smoke_vllm.sh
```

The server command inside the container:

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
| schema | `scripts/check_recipe.py` | 2026-09-11 |
| digest | image sha256 matches; `/model` is the `df42c109` revision, byte-verified | 2026-09-11 |
| health | `/v1/models` 200; log shows `Available KV cache memory: 10.05 GiB`, `Loaded 210 configs`; `smoke_vllm.sh` 5/5 | 2026-09-11 |
| quality | smoke (arith 323, count 1–60, prose, parsed `tool_call`, thinking→`36`); **Hermes harness 10/10** tool calls with correct answers and side effects ([harness-summary.md](results/2026-09-12-v12-1M-k5-off60-util97/harness-summary.md) (v12 re-run 2026-09-12, 10/10)). No requant, so no divergence gate. | 2026-09-11 |
| performance | `knee.sh` C1 within 5% of 89.2 tok/s warm vs a same-window v11 control (79.5) | 2026-09-12 |

**On instruments.** `knee.sh` runs each concurrency twice and the two runs agree within ~1%; it is the decision metric. `replay.py` replays 24 real agent turns and is useful as a workload-shaped smoke test, but at temperature 0 it swings **±17 tok/s** run to run (batched MoE + speculative decode are not bit-deterministic). We published an overnight "win" on it and retracted it the next morning when the knee said the opposite. Bench on the tight instrument.

## Results

### v18 — Many Seat (current)

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

See `limits:` in [`recipe.yaml`](recipe.yaml). The short version: decode is C2C + kernel bound and the offloaded-expert byte count is the lever (~0.76 tok/s per GiB moved into HBM; v12 spends the last HBM headroom on it and pays in KV); quote the tok/s for your content class, not the best one; the autotune hash trap is real; `reasoning_effort: medium` is rejected by the template (use `low|high|xhigh|max`); ~110 GB of host shmem is unaccounted for and the host has no room for a second big model.

## What didn't work

[`research/failure-ledger.md`](research/failure-ledger.md) — SGLang at 3.3 tok/s, four vLLM UVA failures before the first bind, the offload bracket, the autotune misses, a first overnight of decode experiments (THP, unpinned host memory, `--language-model-only`, Rust frontend) that produced one real lesson about instruments and zero adopted flags, and a second night (k-sweep, util-only, off60) that produced v12 — see [`night-two-ledger.md`](results/2026-09-12-v12-1M-k5-off60-util97/night-two-ledger.md). A third night (2026-09-18/19) closed the k=3 mid-band schedule and on-line adaptive placement and left the nightly-image cache-load question open — [`2026-09-19-overnight-ksched-nightly-adaptive/`](results/2026-09-19-overnight-ksched-nightly-adaptive/README.md).

## Rollback

Every bound config is kept as a stopped container `dsv41-vllm-<tag>`. `docker stop` the lane, `docker start` the previous one; ~4 min. Production surfaces on other ports are never part of this lane.

## Credits

Campaign design, briefs, win bars, and verdicts: **Milo** (James Meadlock's Handler agent, Hermes `milo` profile, session model anthropic/claude-fable-5.1) with James. Worker runs (E1–E5 windows, recipe PRs): Grok 4.6 via xAI and claude-fable-5.1 via Nous, from Milo's written briefs; every number here was read back from the worker's JSON. The pin-hot-experts idea: James, in conversation with Grok.

DeepSeek for the model and the in-checkpoint DSpark drafter. The vLLM team for the day-0 image, UVA offload backend, and DeepSeek-V4.1 support. Community data points that shaped decisions: 0xSero and Tech2Wild (4× RTX PRO 6000 and DGX Spark builds), Fraser Price (`dspark-vllm`), LMSYS (Engram huge-page finding). Blog write-up: [al-engr.com](https://al-engr.com/gb300-deepseek-flash-41-testing.html).
