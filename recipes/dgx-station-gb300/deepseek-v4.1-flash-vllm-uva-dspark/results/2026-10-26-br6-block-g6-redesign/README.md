# 2026-10-26 — Card BR6: block verification with the redesigned G6 (G6 PASS; FAIL on shell_ops G2/G2b); block verification RETIRED for cause on this lane, series closed

One candidate was bracketed against this recipe's reference, `v28-prob`, for the third time:

- **A / A2** (controls) = v28-prob = v26 + `"draft_sample_method":"probabilistic"`, rendered byte-identical to the
  stopped-and-kept `dsv41-vllm-1017-v28-prob-BOUND-REF` (G0);
- **B** = v28-prob + `"rejection_sample_method":"block"` (block verification, Sun et al.). This is the BR4 candidate
  "PB" and the BR5 candidate B.

Everything is Card BR5 except **G6**, which was redesigned and pre-registered before launch. The launcher is
`launch_br3.sh`, reused unchanged from Cards BR3–BR5 (sha1 `8066045c`). The image is
`vllm/vllm-openai:deepseekv41-flash-0909` (`sha256:00d577a6…`), with hook v26ar.2 and seed map
`rowmap-mixed-ar-v1.json`. UVA offload is 60 GiB, context 1M, 24 seqs, DSpark k-schedule `[[1,8,5],[9,24,1]]`. The
runner and verdict are BR5's plus the G6 change
([`scripts/DIFF-br6-runner-vs-br5.txt`](scripts/DIFF-br6-runner-vs-br5.txt),
[`scripts/DIFF-br6-verdict-vs-br5.txt`](scripts/DIFF-br6-verdict-vs-br5.txt)). Every other instrument, including
`replay_turns.py` @`5f7934a9`, has the same sha1 on the box as in BR5 ([`instruments.txt`](instruments.txt)).

## The G6 redesign (pre-registered)

[`PREREG-BR6.md`](PREREG-BR6.md) was written at 14:13:15 CDT, before launch. It points to the G6 spec in the desk
brief and makes one change before launch: **8 passes per arm instead of 4**. The brief stays on the box (see
"Withheld"). Its spec, quoted:

> 1. **Passes.** Each arm runs `replay_turns.py` **4 times** (`PASSES=1` each, tags `BR6-<arm>-p1..p4`). Each pass
>    writes its own raw file `<arm>/turns-BR6-<arm>-p<k>.json` and summary `turns-BR6-<arm>-p<k>-summary.json`. The
>    instrument (`m3/replay_turns.py` @5f7934a9) is unchanged; the runner simply invokes it 4 times. If an arm has more
>    than 4 passes, the first 4 are used. If it has fewer than 4, G6 reports **INSUFFICIENT PASSES**: on B that is a
>    FAIL, on a control it is INVALID.
> 2. **Statistic.** The per-arm **mean** of tool-turns emitted per pass, out of 12.
> 3. **Bar.** B passes G6 when both of these hold:
>    - mean(B) lies within **[min(pooled 8 control passes) − 1, max(pooled 8 control passes) + 1]**
>    - **|mean(A) − mean(A2)| ≤ 1.5**. If the two controls differ by more than that, G6 is **INVALID-CONTROL(A,A2)**.
>      The card suffix then becomes `BR6-INVALID-CONTROL-A-A2-G6`, not a FAIL.
> 4. **Reported, not gated:** B's pass-level min and max; the pooled control distribution, as a histogram; each arm's
>    mean; a **per-transcript agreement table**: for each of the 12 tool transcripts, a 1/0 string per arm across its
>    passes. A transcript is flagged `B-CONSISTENT-FLIP` when B is identical on all its passes and every control pass is
>    the opposite. Otherwise it is `stable` or `wobble`. […] This table is an annotation, not a gate.
> 5. **Errors.** Any request error in a B pass is a FAIL (hard gate). Any error in a control pass makes that control
>    INVALID. […]
> 6. **No other changes.** G0–G5, G7, every bar, every cell, the launcher, ISOLATE and the restore are BR5 verbatim […]
> 7. **Retirement rule (carried over from PREREG-BR5).** If B FAILs this G6 *with* consistent per-transcript flips,
>    block verification (PB) is retired on this lane. […]
>
> **Why ±1 on the pooled range and 1.5 on the control means.** I simulated this assuming independent passes, with
> each transcript's emission probability taken from all 14 receipt passes:
> - **Old 2-pass rule:** false-FAIL rate about **33 %**. […]
> - **New rule:** false-FAIL about **0.05 %**, false INVALID-CONTROL about **0.7 %**.
>
> […] The independence assumption is optimistic. BR5 A2's two passes were identical, which suggests passes within a
> boot are positively correlated.

PREREG-BR6 sets the pass count to **8 per arm** (`TURNS_PASSES=8`, `TURNS_REQ=8`), which makes the pooled control set
16 passes. It also adds a hand-applied rule: a `B-CONSISTENT-FLIP` on any transcript is a FAIL even if the script
prints PASS. All other gates and bars are BR5 verbatim, and a trip in any class is a FAIL.

## Run

Bracketed **A (control) → B → A2 (control)**, 2026-10-26 14:13–16:36 CDT. Each arm was one fresh server boot.

- **Adaptive residency.** The hook booted frozen on every arm, then received `GO` +
  `ctrl.json = {"freeze": false, "baseline": true}` and 90 s of settle. Adaptive residency was therefore unfrozen on
  all three arms. Swaps during the cells: A 84,342, B 85,632, A2 84,421.
- **Production lane.** The sidecar recipe's v27 pair was dark for the card and firewalled to loopback; 0 foreign
  requests reached any arm. It was restored at 16:36:02, with KV 4,400,013, block in its Args, and its fail-stop
  watchdog re-armed (pid 2863015, PPID 1).
- **Boots.** All three arms were autotune **hash hits** (Loaded 189 / Saved 189, same cache directory, 5–6 s autotune
  span). They bound after 482 / 482 / 472 s and were ready at 484 / 484 / 474 s (A / B / A2).
- **KV.** **2,481,694 tokens on every arm**, the same as the v28-prob boot log. Block verification allocates nothing.

## Verdict

[`VERDICT.txt`](VERDICT.txt), [`verdict.json`](verdict.json): **`FAIL`, suffix `BR6FAIL-G2-G2b`**, pre-registered,
not re-scored.

- **Passed (8 of 10 gates):** G0 binding, G0b foreign, G1 controls, G3 mixed, G4 teacher-forced, G5 gauntlet,
  **G6 replay_turns (redesigned)**, G7 errors.
- **Failed (2 gates, both on one class, shell_ops C1):** **G2** per-class temp 1.0 and **G2b** temp-0 parity.

## Results

| temp 1.0 | A | **B** | A2 | B ÷ mean(A, A2) |
|---|--:|--:|--:|--:|
| accepted tokens per verify step, per-class window | 3.457 | **3.494** | 3.433 | **1.014** |
| accepted tokens per verify step, mixed C1 | 2.100 | 2.123 | 2.100 | 1.011 |
| mixed C1 tok/s (`ar_mixed_t.py`, 3 × 120 s) | 181.31 | **182.56** | 181.03 | 1.008 (+0.77%, bar −2.0%) |

Temperature-1.0 per-class decode tok/s (mean of r1/r2), bars from G2:

| class | C1 A / A2 / **B** | C1 B ÷ ctrl | C8 A / A2 / **B** | C8 B ÷ ctrl |
|---|---|--:|---|--:|
| prose | 221.6 / 213.4 / **220.9** | 1.016 | 483.4 / 488.1 / **523.3** | 1.077 |
| structured | 242.9 / 244.4 / **244.1** | 1.002 | 597.4 / 600.8 / **615.4** | 1.027 |
| code | 277.3 / 278.9 / **293.3** | **1.055** | 634.4 / 636.9 / **646.8** | 1.018 |
| shell_ops | 333.0 / 317.3 / **308.8** | **0.950 — FAIL** (bar −7.2%) | 733.1 / 735.7 / **735.8** | 1.002 |
| tool_json | 285.7 / 288.6 / **284.9** | 0.992 | 762.7 / 789.9 / **765.7** | 0.986 (bar −5.2%) |

Per-class accepted tokens per verify step (B / ctrl), C1 then C8:

| class | C1 | C8 |
|---|---|---|
| prose | 2.86 / 2.82 | 2.90 / 2.83 |
| structured | 3.12 / 3.06 | 3.13 / 3.07 |
| code | 3.99 / 3.84 | 3.97 / 3.84 |
| **shell_ops** | **4.04 / 4.16** | 4.07 / 4.04 |
| tool_json | 3.66 / 3.67 | 3.73 / 3.70 |

shell_ops C1 is the only cell where B accepted fewer tokens per step than the controls.

**Per-position acceptance** (temp-1.0 window, accepted at draft position *i* ÷ drafts):

| position | 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|
| A | 0.787 | 0.589 | 0.447 | 0.351 | 0.284 |
| **B** | **0.784** | **0.596** | **0.458** | **0.363** | **0.294** |
| A2 | 0.785 | 0.587 | 0.442 | 0.343 | 0.275 |

The shape is the same as in BR4 and BR5: no gain at position 1, and a gain that grows to about 0.015 at positions
4–5.

### G6 replay_turns, redesigned — PASS

Tool turns emitted per pass (of 12, temperature 0, `replay_turns.py`, 8 × `PASSES=1`):

| arm | passes 1–8 | mean |
|---|---|--:|
| A | 8, 7, 8, 9, 7, 8, 7, 8 | 7.75 |
| **B** | **8, 9, 8, 5, 8, 6, 6, 6** | **7.00** |
| A2 | 7, 7, 8, 9, 9, 9, 7, 8 | 8.00 |

- **Bar:** the pooled 16 control passes {7: 6, 8: 6, 9: 4} give [7 − 1, 9 + 1] = **[6, 10]**.
- **B:** mean 7.00 is inside the bar. Its pass-level range is 5–9. B errors: 0.
- **Controls:** they agree, |7.75 − 8.00| = 0.25 ≤ 1.5.

Per-transcript agreement (1 = tool call emitted in that pass, passes in order):

| transcript | A | B | A2 | flag |
|---|---|---|---|---|
| 0 | 11111111 | 11111111 | 11111111 | stable |
| 1 | 11110111 | 11101011 | 11111111 | wobble |
| 2 | 00000000 | 01001000 | 00011100 | wobble |
| 3 | 11111111 | 11111111 | 11111111 | stable |
| 4 | 00000000 | 00000000 | 00000000 | stable |
| 5 | 10111111 | 11100001 | 11111111 | wobble |
| 6 | 11111111 | 11111110 | 11111111 | wobble |
| 7 | 11111100 | 11101100 | 00010101 | wobble |
| 8 | 00000000 | 00000000 | 00000000 | stable |
| 9 | 11111111 | 11111111 | 11111111 | stable |
| 10 | 00010001 | 00000000 | 00101000 | wobble |
| 11 | 11111111 | 11111111 | 11111111 | stable |

**No `B-CONSISTENT-FLIP`.** The hand-applied rule from PREREG-BR6 does not fire, and no transcript sits at B 0/8
against 16/16 control hits, so no turn-by-turn diff was needed.

- **Transcript 7** spans the whole range within the controls themselves: A 6/8, A2 3/8, and B 5/8 between them.
- **Transcript 5** is B's largest drift: 4/8 against A 7/8 and A2 8/8. B still emitted on it in four passes, so it is
  not a consistent flip.

Applied to the old 2-pass receipts as information, the new statistic would have put BR4 P and PB inside their bars.
It still puts BR5 B outside its bar (6.5 vs [7, 10]); that is the `BR4/BR5 G6 history` block in `VERDICT.txt`.

### G2 and G2b — FAIL, both on shell_ops C1

**G2 (temperature 1.0).**

- **Rule:** B ≥ **each** control × (1 − bar), where bar = max(2%, 1.5 × |A − A2| / mean(A, A2)).
- **Bar:** shell_ops C1 controls read A 332.98 and A2 317.27 tok/s, a spread of 4.83%. The bar is therefore
  1.5 × 4.83% = **7.25%**.
- **Threshold vs A:** 332.98 × (1 − 0.0725) = **308.85 tok/s**. **B read 308.76 and missed by 0.09 tok/s.**
- **Threshold vs A2:** 294.27 tok/s. B cleared it.
- **B's two repeats:** r1 295.15 tok/s (engine acceptance 0.579), r2 322.37 (0.638). The controls' repeats were A
  338.70 / 327.26 and A2 310.37 / 324.18.
- **Ratio:** B ÷ mean(A, A2) = 0.950.

**G2b (temperature-0 parity).**

- **Why it applies:** greedy rows never use block verification, so B must equal A within noise.
- **Reading:** shell_ops C1 read B 338.79 against a control mean of 350.39 (A 348.54, A2 352.24): **−3.31%**.
- **Bar:** max(2%, 1.5 × the A/A2 difference 1.06% = 1.58%, 1.5 × the largest control r1/r2 spread 0.95% = 1.43%) =
  **2.0%**.
- **Other classes:** all four pass. Prose +1.7% (bar 6.1%), structured +2.2% (5.3%), code −0.3% (3.8%), tool_json
  +6.2% (6.5%).

**Other gates.**

- **G1 controls:** A vs A2 agree within 0.4–4.8% per class at C1. shell_ops t1 C1 is the widest at 4.8%.
- **G4 teacher-forced:** 79,544 positions, 39 docs, 0 token-mismatch docs. Top-1 flips B-vs-A 1.140% and B-vs-A2
  1.075%, against an A2-vs-A floor of 1.082% (bar ≤ 1.5 × floor = 1.624%). Margin-weighted flips at ref p ≥ 0.9 on
  agent/tool docs: 0 on every compare. This check is prefill-only; the drafter and verifier do not run in it.
- **G5 gauntlet:** 12/12 on all three arms.
- **G7:** 0 FF errors and 0 foreign requests on all arms.
- **x3 acceptance canary** (temp 0, inside each window): identical across A/B/A2 on every class.

### shell_ops across the three block-on-probabilistic cards

| card | candidate vs | shell_ops t1 C1 B ÷ ctrl | shell_ops t1 C8 B ÷ ctrl | shell_ops t0 C1 (G2b) | what tripped |
|---|---|--:|--:|---|---|
| BR4 | PB vs v26 controls (greedy drafting) | 1.126 | 1.157 | **+3.6% vs bar 2.6% — FAIL** | G2b shell_ops, G6 |
| BR5 | B vs v28-prob | 1.039 | **0.985** (bar −2.0%; lowest cell of the card) | +2.4% vs bar 10.3% | G6 only |
| BR6 | B vs v28-prob | **0.950 — FAIL** (bar −7.2%) | 1.002 | **−3.3% vs bar 2.0% — FAIL** | G2 + G2b shell_ops |

Both speed or parity trips in the series were on shell_ops: BR4 G2b and BR6 G2/G2b. In BR5 the trip was G6, and
shell_ops C8 was the only class below 1.0. The two G2b trips point in opposite directions (+3.6%, then −3.3%). Our
reading, untested, is that shell_ops, the class with the highest acceptance on this lane (about 4.1 tokens per step),
is the most sensitive to run-to-run variation in this bracket. Block verification's gain of +1.4–2.5% tokens per step
is smaller than that variation.

## Decision (James, 2026-10-26 ~16:45 CDT)

**Block verification is RETIRED for cause on the GB300-only lane, and the series is closed.**

- **No clean pass.** Three brackets of block on probabilistic drafting (BR4 PB, BR5, BR6) never produced one.
- **The gain is small and inside the noise.** Block's gain on this lane is +1.4% (BR6), +1.5% (BR4, over P) and +2.5%
  (BR5) tokens per verify step. That sits inside the lane's per-class noise: in this card the controls alone differ by
  up to 4.8% per class at C1, and B's own shell_ops repeats differ by 8.8%.
- **The G6 redesign stays.** It is kept as the standard G6 for this lane: 8 passes per arm, mean statistic, pooled
  control bar ±1, a control-agreement check, and the per-transcript table. The old 2-pass rule false-fails about 33% of
  the time under independence; the new one about 0.05%.
- **The reference stays v28-prob.** It is unchanged. `dsv41-vllm-1017-v28-prob-BOUND-REF` is stopped-and-kept, and
  v26seed is its rollback.
- **The sidecar sibling is unaffected.** Its v27 (+ block) passed clean on its own lane (Card BR1,
  [bundle](../../../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/README.md)).
  That lane already drafted probabilistically and uses a different k-schedule. `:30006` keeps serving v27.

Earlier cards: [BR5](../2026-10-26-br5-block-rebracket/README.md) (FAIL-G6),
[BR4](../2026-10-26-br4-probabilistic-drafting/README.md) (P PASS → v28-prob; PB FAIL-G2b-G6),
[BR3](../2026-10-26-br3-block-verification-null/README.md) (block on greedy drafting, a null). Write-up:
[al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html).

## Files

| path | what it is |
|---|---|
| `PREREG-BR6.md` | the pre-registration, written before launch (14:13:15 CDT) |
| `VERDICT.txt`, `verdict.json` | the pre-registered scoring (`scripts/br6_verdict.py`), including the G6 per-transcript table and the BR4/BR5 G6 history annotation |
| `SUMMARY.txt`, `runner.log`, `phases.jsonl` | runner timeline: dark, isolate, boots, facts, GO/unfreeze, cells (8 replay passes per arm), restore |
| `instruments.txt` | sha1 of every script and fixture as run **on the box** (the published copies differ by the path scrub only), image digests, A/B `SPEC_EXTRA`, GO ctrl, sampling |
| `launch-*.txt`, `health-*.json`, `errcount-*.txt`, `oom-*.txt`, `mem.tsv`, `settle-pre-*.txt`, `preflight-*.txt`, `kv-ref.txt` | launcher output, chat health, error-line counts, OOM counts, MemAvailable samples, pre-boot host state, the v28-prob KV reference |
| `raw/<arm>/` | `facts.json`, `boot-facts.log`, `boot-autotune.log`, `boot-block.log`, `boot-prob.log`, `boot-pin.log` (hook rehome lines), `args.json` / `env.json` / `binds.json` (from `docker inspect`), `go.json` + `go.log` + `ar-ctrl.json` + `ar-lines.tsv` (adaptive GO/unfreeze and swap/telemetry line counts per cell), `perclass-t1/` and `perclass-t0/` (tables md+json; **per-request `rows` stripped**, `rows_withheld` set), `mixed-*.json.gz` (per-request timings only; synthetic prompt pools), `specmetrics/` (`vllm:spec_decode*` counters per window, including the per-position counters and one start/end pair per replay pass), `turns-BR6-<arm>-p<k>-summary.json` (8 per arm: tool/text turns emitted, emitted tool-transcript indices, errors, wall) and `turns-p<k>.log` |
| `raw/tf-compare-*.json`, `raw/tfcmp-*.log`, `raw/flipsm-*.txt` | teacher-forced compares and margin-weighted flips |
| `raw/gauntlet-<arm>/` | gauntlet summary + sandbox isolation check (fixture inputs and per-task jsonl omitted) |
| `raw/production/` | production lane (sidecar v27) args/env before and after the card, plus the v28-prob reference args/env/binds that G0 compared against |
| `raw/restore-v27-block.facts.log` | filtered KV / autotune / startup lines from the production restore boot |
| `scripts/` | `cardBR6_runner.sh`, `br6_verdict.py`, `DIFF-br6-runner-vs-br5.txt`, `DIFF-br6-verdict-vs-br5.txt`, plus `launch_br3.sh` and `DIFF-br3-launch.txt` reused from BR3 (same sha1 `8066045c`), and the BR1 instruments it reuses unchanged (`perclass_run_t.sh`, `perclass_decode_t.py`, `ar_mixed_t.py` and their `DIFF-*.txt`; same sha1 on the box as in the BR1/BR3/BR4/BR5 bundles), exactly as run, with box paths scrubbed |
| `throughput.csv` | every speed / acceptance / per-position / KV / flip cell with its receipt |

This bundle was built with the same copy-and-scrub transform as the BR5 bundle. Re-run against the box receipts on
2026-10-26, that transform reproduces 288 of the 289 files in the committed BR5 bundle byte for byte, including its
`throughput.csv`. The 289th file is that bundle's hand-written README.

## Withheld, on purpose

- **Per-class prompts.** `perclass_prompts.jsonl` (sha1 `618898aac432`) is real Hermes transcripts and stays
  **private**. The tables are published verbatim. The per-request `rows` (output heads) are removed from every
  perclass/agentfix JSON.
- **Replay passes.** The `turns-BR6-<arm>-p<k>.json` files replay `transcript_fixture.json`, which is also private
  transcripts, and carry per-turn calls and content. Only the per-pass summaries are published.
- **Teacher-forced dumps.** The per-position dumps (`tf-*.jsonl`) stay on the box; the compares are published.
- **Other box receipts.** These stay on the box: server post-cell logs (`postlog-*`), full boot logs, the hook's
  `swaps.jsonl` / `telemetry.jsonl` / `steps.bin`, `preflight-docker-ps-a.txt`, `verdict.err`, the gauntlet per-task
  jsonl and fixture trees, and the full restore log.
- **Desk material.** The brief and dry-run fabricator (`BRIEF-BR6.md`, `dryrun/`) stay on the box; the dry-run numbers
  are synthetic by design. The G6 spec above is quoted from the brief.
- **Path scrub.** The box home directory → `~`, LAN address → `<gb300>`.
