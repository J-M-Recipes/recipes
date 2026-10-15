# DFlash2 on the GLM-5.3-Flash Spark TP2 lane — weekend campaign, 2026-10-03/05

Written: `2026-10-05T05:55-05:00` on the M4. Raw per-cell JSON, launchers, diffs, boot greps, and runner logs for every boot are on Spark1 under `~/glm53-e4-e13-20261003/`, `~/glm53-diag-20261004/`, `~/glm53-k-20261004/`, `~/glm53-cq-20261004/`, mirrored here under `diag/`, `k/`, `cq/` (E4/E13 are in `../dflash2-e4-e13-20261003/`). `cq/campaign-summary.json` is the machine-readable rollup of every arm below.

## Verdict in one paragraph

On this pair the incoai `7d74cdd` DFlash2 drafter accepts **~2.3 tokens per verify step on open prose, ~3.3–3.7 on code, 7.9 on count-style structured output, 6.5 on a one-line tool call** — and it does that identically across every axis we changed: drafter revision, CUDA graphs, sampling, thinking regime, prompts, K, Tony's overlay vs canada-quant's patched fork, NVFP4 vs W4A16 target. Nothing is broken. Prose is high-entropy and the block tail dies after position 2 (`0.65 · 0.37 · 0.21 · 0.08 · 0.04 · 0.01 · 0.00`). The "3.6" headline numbers from GB300 and canada-quant come from code/math-heavy fixtures where our per-position curve matches theirs to the second decimal. The one lever that moved real throughput was **K**: dropping from 7 to 4 or 3 stops paying verify cost for dead positions and gives +11–16 % at C1 prose and +25–40 % at C4/C8, where K=7 was actually *losing* to plain autoregressive decoding.

## Arms

All boots: Tony eager recipe (`--enforce-eager`, TP2 over bond0/RoCE, `--gpu-memory-utilization 0.85`, `--max-model-len 262144`, `--max-num-seqs 6`, `--block-size 2304`, fp8_e4m3 KV with a 3 GiB pin, `--default-chat-template-kwargs enable_thinking=false`), drafter `incoai/GLM-5.3-Flash-DFlash2@7d74cdd` (safetensors `8931dc52…`, config `c4aeac01…`) unless stated. Kernel `6.17.0-1031-nvidia` both nodes, clocks 208/3003 MHz at every launch and teardown, no `-lgc`.

| Arm | Image | Target | K | Boot → ready | Dir |
|---|---|---|---|---|---|
| AR | `funland/glm53-dflash2:ccc5a78e` (`5be2a57f…`) | LibertAIDAI NVFP4 `aa28e1f` | none (no `--speculative-config`) | 03:47:44 → 04:04:54 (17 min) | `diag/raw/ar` |
| TONY_K7 | same | NVFP4 | 7 | 04:12:03 → 04:31:32 (19.5 min) | `diag/raw/dflash` |
| TONY_K4 | same | NVFP4 | 4 | 07:28:19 → 07:47:32 | `k/raw/k4` |
| TONY_K3 | same | NVFP4 | 3 | 07:52:09 → 08:11:38 | `k/raw/k3` |
| TONY_K7 regimes (G1) | same | NVFP4 | 7 | 09:37:33 → 09:57:32 | `k/raw/g1` |
| CQ_K7 | `ghcr.io/canada-quant/vllm-glm53-flash-sm121:v2-w4a16-dflash2e` (`f42a8a52…`, vLLM `0.1.dev20051+g487ecf187`, patches `8a3ecfb0…` + `b894ad44…` verified in-image) | NVFP4 | 7 | 21:12:56 → 21:33:12 (20 min) | `cq/raw/cq`, `cq/raw/cq-g1` |
| W4A16_K7 | canada-quant image | `canada-quant/GLM-5.3-Flash-W4A16-MTP@5681f7c` (shas in `cq/w4a16-sha256.txt`, verified on both nodes), `--moe-backend marlin` dropped (engine chose `MARLIN WNA16 MoE` itself) | 7 | 05:22:55 → 05:33:25 (**10.5 min**) | `cq/raw/w4a16`, `cq/raw/w4a16-g1` |

E4 (greedy vs T=1.0/top-p 0.95) and E13 (Hub `main` `bf582e4` drafter) are written up separately in `../dflash2-e4-e13-20261003/RESULTS.md`: both NO_PASS, sampling −3 % accept / −5 % tok/s, main +3 % / +2 %, server ignores `seed`.

Every arm booted first try. No rank died. No HTTP errors in any cell. The "non-7 K wedges DFlash2" claim from the canada-quant card did not reproduce: K=4 and K=3 booted and served cleanly on this image.

## Matrix (E1 cells + GB300 probe prompts), aggregate tok/s / acceptance length

Greedy, thinking off, stream false, 300 tokens (probe rows 512), C1 prose is three reps. Acceptance length = completion_tokens / drafts delta from `/metrics`.

| cell | AR | TONY_K7 | TONY_K4 | TONY_K3 | CQ_K7 | W4A16_K7 |
|---|---:|---:|---:|---:|---:|---:|
| prose_c1_r1 | 15.2 | 19.0 / 2.29 | 23.2 / 2.29 | 25.3 / 2.34 | 20.1 / 2.36 | 20.4 / 2.29 |
| prose_c1_r2 | 15.1 | 20.7 / 2.42 | 21.6 / 2.17 | 23.1 / 2.13 | 19.9 / 2.33 | 20.5 / 2.31 |
| prose_c1_r3 | 15.2 | 19.9 / 2.31 | 22.0 / 2.17 | 22.4 / 2.05 | 22.1 / 2.59 | 20.8 / 2.33 |
| prose_c4 | 38.0 | 31.9 / 2.38 | **51.4** / 2.22 | 43.4 / 2.27 | 30.8 / 2.38 | 30.6 / 2.19 |
| prose_c8 | 42.9 | 35.9 / 2.37 | 45.0 / 2.23 | **50.9** / 2.20 | 36.3 / 2.41 | 34.6 / 2.22 |
| structured_c1 | 15.0 | 64.4 / 7.89 | 46.7 / 5.00 | 40.4 / 4.00 | 63.0 / 7.89 | 64.6 / 7.69 |
| structured_c4 | 50.2 | 66.0 / 7.89 | 126.7 / 5.00 | 101.5 / 3.97 | 64.4 / 7.89 | 67.1 / 7.89 |
| structured_c8 | 50.6 | 116.7 / 7.89 | 123.6 / 4.93 | 107.5 / 3.97 | 117.9 / 7.89 | 121.0 / 7.77 |
| agent_c1 | 1.7 | 1.8 / 6.50 | 1.8 / 4.33 | 1.8 / 4.33 | 1.9 / 6.50 | 1.8 / 6.50 |
| agent_c4 | 18.4 | 6.2 / 6.50 | 21.7 / 4.33 | 22.3 / 4.33 | 6.1 / 6.50 | 6.3 / 6.50 |
| agent_c8 | 10.7 | 27.5 / 6.50 | 29.9 / 4.33 | 11.5 / 4.33 | 24.8 / 6.50 | 29.2 / 6.50 |
| probe_prose0 | 15.2 | 18.5 / 2.12 | 20.4 / 2.01 | 22.6 / 2.07 | 17.8 / 2.06 | 19.8 / 2.20 |
| probe_prose1 | 15.2 | 22.2 / 2.65 | 25.9 / 2.61 | 24.9 / 2.34 | 23.3 / 2.78 | 23.4 / 2.68 |
| probe_code0 | 15.2 | 30.0 / 3.63 | 29.5 / 2.99 | 30.6 / 2.89 | 30.5 / 3.71 | 29.8 / 3.46 |
| probe_code1 | 15.2 | 27.4 / 3.35 | 27.6 / 2.83 | 30.7 / 2.93 | 24.8 / 3.01 | 28.0 / 3.26 |

C1 prose medians: AR **15.16**; K7 **19.87** / 2.31; K4 **22.04** / 2.17; K3 **23.11** / 2.13; CQ **20.14** / 2.36; W4A16 **20.45** / 2.31.

Reading notes:
- AR is flat at 15.2 on every fixture — the target decodes one token per ~66 ms regardless of content. DFlash2 K7 at 2.3 tok/step and 19.9 tok/s is ~116 ms/step: verifying 8 positions plus the draft pass costs ~50 ms over an AR step, and prose only pays back 2.3 of them.
- Agent tok/s is prefill-dominated (13 completion tokens); the agent number is acceptance length. agent_c4/c8 wall times swing with warmth and are not decode measurements.
- Agent acceptance 6.5 at K7 and 4.33 at K4/K3 is the same 13-token tool call over 2 vs 3 drafts: the call fits one block at K7. Not a regression.
- Structured drops with K because the ceiling is K+1 by definition. At C8 K4 still beats K7 (123.6 vs 116.7).

## Per-position acceptance (probability a verify step keeps draft position k)

| arm | fixture | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TONY_K7 | prose r3 | 0.62 | 0.36 | 0.19 | 0.07 | 0.05 | 0.01 | 0.00 |
| CQ_K7 | prose r3 | 0.72 | 0.45 | 0.26 | 0.10 | 0.03 | 0.01 | 0.00 |
| W4A16_K7 | prose r3 | 0.67 | 0.39 | 0.20 | 0.04 | 0.02 | 0.00 | 0.00 |
| TONY_K4 | prose r3 | 0.62 | 0.33 | 0.16 | 0.05 | | | |
| TONY_K3 | prose r3 | 0.62 | 0.31 | 0.12 | | | | |
| TONY_K7 | code0 | 0.78 | 0.57 | 0.40 | 0.31 | 0.24 | 0.19 | 0.13 |
| CQ_K7 | code0 | 0.79 | 0.55 | 0.43 | 0.33 | 0.25 | 0.20 | 0.17 |
| W4A16_K7 | code0 | 0.78 | 0.52 | 0.42 | 0.30 | 0.18 | 0.16 | 0.12 |
| canada-quant, incoai drafter, their 500-prompt holdout, 8× B300 (their card) | mixed | 0.75 | 0.55 | 0.41 | 0.32 | 0.25 | 0.20 | 0.16 |

Positions 4–6 on prose contribute ~0.07 tokens per step combined. That is what K=7 was paying ~40 ms per step to verify. On code the tail is alive and K=7 is the right setting — the `probe_code*` rows lose 0.4–0.7 acceptance at K3/K4 with no tok/s gain.

## Decode regime (G1), prose accept median by regime, same server

| regime | TONY_K7 | CQ_K7 | W4A16_K7 |
|---|---:|---:|---:|
| thinking off, T=0 | 2.42 | 2.31 | 2.25 |
| `reasoning_effort: low` (**ignored by this template — `think=False`**) | 2.35 | 2.48 | 2.41 |
| thinking on, T=0 | 2.63 | 2.83 | 2.55 |
| thinking on, T=1.0 / top-p 0.95 (canada-quant protocol) | 2.46 | 2.47 | 2.46 |

Thinking on buys ~0.2–0.4 because reasoning tokens draft a little easier. Nothing reaches 3. Note the identical greedy request produced different acceptance on a repeat (code0 3.63 vs 4.61 in one pair of rows): acceptance on this lane is not run-to-run deterministic (argmax tie-breaking in the verify kernel, already noted on the GB300 recipe), so single numbers carry ±0.2–0.3 noise. Medians of 3 are the minimum for any verdict.

## What the comparisons actually established

1. **GB300 "2.77–3.95 on prose" and Spark "2.3" are the same drafter on the same bytes** (`LibertAIDAI/GLM-5.3-Flash-NVFP4@aa28e1f` + `7d74cdd` on both). The GB300 figure was a 4-prompt probe whose two prose prompts sit at 2.1–2.8 here too; its upper end is the code prompts. There is no SM121 acceptance gap to chase.
2. **canada-quant's 3.58–3.68** is a code/math/STEM-heavy holdout with thinking traces. Our code per-position curve matches their incoai row within 0.03 at every position. Their W4A16 target on their image with the incoai pin gave 2.31 prose here — identical to NVFP4.
3. The two canada-quant fork patches are a >24K-context top-k crash fix and full-attention-drafter KV support. They do not touch short-context acceptance, and the CQ_K7 boot confirmed it (2.36 vs 2.31).
4. Thinking regime, sampling, prompts, drafter revision (E13), CUDA graphs (E1): each within noise.

## Memory

Lowest head MemAvailable during any C8 cell: 8.3 GiB (K3 agent_c8). W4A16 ran heavier at ~15.3–15.9 GiB head / 20.2–20.7 GiB worker because its pack-quantized tree holds more host-side state; AR ran at 11.2–12.1 GiB. Fail bar (4 GiB) never approached. MemAvailable recovered to ~117 GiB on both nodes after every `docker rm -f`, no drop_caches, no reboot.

## Operational notes worth keeping

- `pkill -f <script-name>` over SSH killed the SSH session that contained the pattern (exit 255). Kill by PID.
- The first `hf download` of the 191 GB W4A16 tree silently stalled at 00:17 and sat for 4.5 h with one dead TCP connection; the client never retried. The re-pull under a 5-minute stall watchdog (`cq/fetch-w4a16.sh`) finished in 11 min, then `rsync -aH` to Spark2 over RoCE took 9 min and `sha256sum -c` passed 13/13. Keep the watchdog pattern for any multi-shard pull.
- `du` double-counts hardlinked HF cache blobs: the `models--LibertAIDAI--GLM-5.3-Flash-NVFP4` cache dir on Spark2 shares inodes with the served tree and is not reclaimable space.
- Deleted on James's approval 2026-10-04 22:08: Spark1 `models/nemotron-3-super-120b-bf16`, `nemotron-weights`, `nemotron-offload`, `nemotron-hf` (→ 868 G free); Spark2 `hub/models--lukealonso--MiMo-V2.5-NVFP4` and the dangling `models/MiMo-V2.5-NVFP4` symlink (→ 248 G free, now 70 G with W4A16 in place).
- W4A16 cold-binds in 10.5 min vs ~20 for NVFP4 on the same image — half the restart cost. Not measured for AR speed or long-context behaviour.
- End state 2026-10-05 05:52: no rank containers on either node, port 8888 free, DS4-F / k3-g2a containers still Exited as found, Aug 27 canary sha `77c99b37…` and 7d74cdd pin untouched, Spark1 690 G free, Spark2 70 G free.

## Suggested next plans for the recipe

Ordered by evidence-per-hour. Each is one axis, one boot, pass bar stated.

1. **Adopt K=4 on the Tony NVFP4 lane (recommended default).** Measured: C1 prose +11 %, C4 prose +61 %, C8 +25 %, agent unchanged, structured C1 −27 % (46.7 vs 64.4, still 3× AR). K=3 is +16 % / +14 % / +42 % with a larger structured hit. Hermes decode is prose and tool calls, so K=4 is the right trade; K=7 is only right for code-heavy serving. Change: one integer in `launch-dflash-rank.sh`. Gate before cutover: a 30-minute soak at C4 mixed with `MemAvailable` ≥ 6 GiB and the 6.5/4.33 agent acceptance holding. Rollback: K=7 script untouched on disk.
2. **Measure W4A16 as a lane, not as a drafter axis.** It booted in half the time, holds the same acceptance, and canada-quant serves it with CUDA graphs and EP at `gmu 0.795` with an 8 GiB KV pin (366K-token pool). Unknowns that decide it: AR tok/s on W4A16 vs NVFP4 (quant kernel speed), long-context retrieval through 98K like the Aug 27 canary, tool-call fidelity, and whether its larger host footprint (+6 GiB) matters at C6. One boot each for AR and DFlash2-K4 on W4A16 with the full E1 matrix plus a 50K needle. Pass: C1 prose ≥ NVFP4 K4 (22.0), needle correct, MemAvailable ≥ 6 GiB. Costs nothing new: weights are on both nodes now.
3. **Graphs on K=4.** E1 showed graphs did not help at K=7 because the step was dominated by the 8-position verify. At K=4 the step is shorter and launch overhead is a larger fraction. One boot: K=4 without `--enforce-eager`, same matrix. Pass: C1 prose +10 % over K4 eager, acceptance within 10 %, and `per_pos` counters not pinned at 1.00 (the #53030 collision check). This is the cheapest remaining speed experiment with a real mechanism behind it.
4. **Retire the "SM121 acceptance gap" line of work.** Do not run canada-quant's `-G` drafter, drafter TP=1, or any further target/overlay swaps expecting acceptance to move. The curve is the drafter's, not the lane's.
5. **Drafter training, scoped honestly.** Only lever left for prose acceptance, with a modest ceiling: Inco's own best chat number on 4× GB300 is 4.19 vs 5.9 on math, and canada-quant's 736K-sample run beat the incoai reference by 0.044. A drafter trained on our thinking-off prose/agent mix might lift prose from 2.3 to ~2.6–2.8, which at K=4 is worth maybe +10 % tok/s. Plan as previously drafted (AngelSpec + vLLM `extract_hidden_states`, 5-tap incoai geometry for drop-in, target served by vLLM on the GB300, prompts from public sets plus Hermes traces, completions regenerated by the target, Apache release) — ~4–5 days of GB300 exclusive. Worth doing for the Apache license and the teaching write-up; not worth doing for speed alone. Gate: James's call on taking `:30001`/`:30006` down.
6. **Fix the regime knob.** `reasoning_effort` is silently ignored by `chat_template_mm.jinja`; only `enable_thinking` works. Any client (Hermes `spark-ds4`-style provider block, Echo) that sends `reasoning_effort` to this lane is getting thinking-off. Worth a one-line template check before anyone benchmarks "low effort" again.
7. **Publish.** The numbers that belong on the al-engr GLM page: AR 15.2 / DFlash2 K7 19.9 / K4 22.0 / K3 23.1 C1 prose; K7 loses to AR at C4–C8 on prose, K4 wins; structured 7.9-of-8 ceiling; per-position curve; "acceptance on open prose is ~2.3 for this drafter on every stack we tried, GB300 included". Credit canada-quant for the image, patches, and the W4A16 quant; Inco for the drafter (CC BY-NC-ND, research/eval only); Tony for the lane.
