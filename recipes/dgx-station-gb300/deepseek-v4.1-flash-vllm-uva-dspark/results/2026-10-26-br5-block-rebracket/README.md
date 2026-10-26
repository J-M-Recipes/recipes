# 2026-10-26 — Card BR5: block verification re-bracketed against v28-prob (FAIL-G6: an instrument finding); PB retired pending BR6

One candidate was bracketed against this recipe's reference, `v28-prob`:

- **A / A2** (controls) = v28-prob = v26 + `"draft_sample_method":"probabilistic"`, rendered byte-identical to the
  stopped-and-kept `dsv41-vllm-1017-v28-prob-BOUND-REF` (G0);
- **B** = v28-prob + `"rejection_sample_method":"block"` (block verification, Sun et al.) — the BR4 candidate "PB",
  now re-bracketed alone as BR4 asked.

Everything else is the v28-prob lane exactly as booted: `launch_br3.sh` (reused unchanged from Card BR3/BR4, sha1
`8066045c`), image `vllm/vllm-openai:deepseekv41-flash-0909` (`sha256:00d577a6…`), hook v26ar.2, seed map
`rowmap-mixed-ar-v1.json`, UVA offload 60 GiB, 1M context, 24 seqs, DSpark k-schedule `[[1,8,5],[9,24,1]]`. The
launch delta is the BR3 one, [`scripts/DIFF-br3-launch.txt`](scripts/DIFF-br3-launch.txt); each arm's `SPEC_EXTRA` is
in [`instruments.txt`](instruments.txt). The runner and verdict are BR3's with the arms and reference swapped
([`scripts/DIFF-br5-runner-vs-br3.txt`](scripts/DIFF-br5-runner-vs-br3.txt),
[`scripts/DIFF-br5-verdict-vs-br3.txt`](scripts/DIFF-br5-verdict-vs-br3.txt)); every other instrument has the same
sha1 on the box as in BR4.

**Pre-registered before launch** ([`PREREG-BR5.md`](PREREG-BR5.md), 11:33:05 CDT): gates and bars = Card BR3/BR4
verbatim; a trip in **any** class is a FAIL; "REPRODUCED" = the same cell trips again in the same direction
(annotation only); **if G6 fails again, PB is retired on this lane until a third design with more replay passes
exists.**

Bracketed **A (control) → B → A2 (control)**, 2026-10-26 11:33–13:52 CDT. Each arm was one fresh server boot. The
hook booted frozen on every arm, then got `GO` + `ctrl.json = {"freeze": false, "baseline": true}` and 90 s of
settle, so adaptive residency ran unfrozen on all three arms (`raw/<arm>/go.json`, `raw/<arm>/ar-lines.tsv`). The
production lane (the sidecar recipe's v27 pair) was dark for the card and firewalled to loopback; 0 foreign requests
reached any arm. It was restored at 13:51:47 (KV 4,400,013, block in Args) with its fail-stop watchdog re-armed
(pid 2801642, PPID 1).

All three arms were autotune **hash hits** (Loaded 189 / Saved 189, same cache dir, 5–6 s autotune span), bound
after 471 / 472 / 472 s and ready at 474 / 474 / 475 s (A / B / A2). **KV 2,481,694 tokens on every arm** (= the
v28-prob boot log; block verification allocates nothing).

## Verdict

[`VERDICT.txt`](VERDICT.txt), [`verdict.json`](verdict.json): **`FAIL`, suffix `BR5FAIL-G6`**, pre-registered, not
re-scored. Nine of ten gates pass (G0 binding, G0b foreign, G1 controls, G2 per-class temp 1.0, **G2b temp-0 parity**,
G3 mixed, G4 teacher-forced, G5 gauntlet, G7 errors). **G6 replay_turns fails.**

## Results

| temp 1.0 | A | **B** | A2 | B ÷ mean(A, A2) |
|---|--:|--:|--:|--:|
| accepted tokens per verify step, per-class window | 3.417 | **3.513** | 3.437 | **1.025** |
| accepted tokens per verify step, mixed C1 | 2.100 | 2.124 | 2.100 | 1.012 |
| mixed C1 tok/s (`ar_mixed_t.py`, 3 × 120 s) | 180.95 | **182.54** | 180.98 | 1.009 (+0.87%) |

Temperature-1.0 per-class decode tok/s (mean of r1/r2), bars from G2:

| class | C1 A / A2 / **B** | C1 B ÷ ctrl | C8 A / A2 / **B** | C8 B ÷ ctrl |
|---|---|--:|---|--:|
| prose | 218.7 / 220.0 / **222.1** | 1.013 | 494.7 / 492.6 / **513.9** | 1.041 |
| structured | 239.4 / 244.1 / **247.8** | 1.025 | 604.9 / 590.1 / **615.2** | 1.030 |
| code | 271.4 / 273.1 / **286.8** | **1.054** | 621.2 / 656.9 / **670.7** | 1.050 |
| shell_ops | 321.2 / 325.2 / **335.7** | 1.039 | 724.2 / 726.0 / **714.4** | 0.985 (bar −2.0%) |
| tool_json | 282.6 / 292.7 / **292.4** | 1.017 | 768.2 / 762.4 / **771.2** | 1.008 |

Per-class accepted tokens per verify step (B / ctrl): prose C1 2.92 / 2.82, C8 2.93 / 2.81; structured 3.14 / 3.04
and 3.12 / 3.04; code 3.91 / 3.77 and 3.98 / 3.92; shell_ops 4.28 / 4.12 and 4.01 / 4.07; tool_json 3.68 / 3.66 and
3.73 / 3.60.

**Per-position acceptance** (temp-1.0 window, accepted at draft position *i* ÷ drafts):

| position | 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|
| A | 0.781 | 0.584 | 0.439 | 0.341 | 0.272 |
| **B** | **0.787** | **0.602** | **0.462** | **0.365** | **0.297** |
| A2 | 0.783 | 0.586 | 0.443 | 0.347 | 0.278 |

Block verification adds 0.005 at position 1 rising to 0.022 at position 5 over mean(A, A2) — the same shape as BR4
(PB over P: +0.001 → +0.012).

- **G1 controls:** A vs A2 agree at C1 within 0.6–3.5% per class at temp 1.0 and 0.6–2.8% at temp 0.
- **G2b temp-0 parity — PASS:** prose −1.5% (bar 12.8%), structured −1.2% (3.7%), code +0.7% (2.7%), **shell_ops
  +2.4% (bar 10.3%)**, tool_json −1.8% (8.1%). shell_ops also sits inside BR4's 2.6% bar, the fixed-bar annotation the
  pre-registration asked for. The BR4 trip (+3.6% vs 2.6%) did **not** reproduce.
- **G4 teacher-forced** (79,544 positions, 39 docs, 0 token-mismatch docs): top-1 flips B-vs-A 1.072%, B-vs-A2
  1.064%, against an A2-vs-A floor of 1.075% (bar ≤ 1.5 × floor). Margin-weighted flips at ref p ≥ 0.9 on agent/tool
  docs: 0 on every compare ([`raw/flipsm-*.txt`](raw/)). This is a prefill-only check of the target model; the drafter
  and verifier do not run in it.
- **G5 gauntlet** 12/12 on all three arms. **G7:** 0 FF errors and 0 foreign requests on all arms.
- **x3 acceptance canary** (temp 0, inside each window): identical across A/B/A2 on every class.
- **Adaptive residency was live during the cells.** Swaps between pct1-start and turns-end: A 82,809, B 82,354,
  A2 82,303.

### G6 replay_turns: the failure, and why it is an instrument finding

`replay_turns.py` replays 12 tool turns from real Hermes transcripts at temperature 0, two passes per arm, and counts
the turns on which the model emits a tool call. The bar is the two controls' four per-pass counts taken as a range.

**BR5: B per pass [6, 7]/12 against a control range of 8–9 (A [9, 8], A2 [8, 8]) — FAIL, in the "fewer" direction.**
BR4's PB trip was [8, 9] against 6–8, in the "more" direction. The pre-registered annotation reads
**NOT-REPRODUCED — opposite-direction trip**.

Every per-pass count from the seven arms of BR4 and BR5 (`raw/<arm>/turns-*-summary.json` here and in the
[BR4 bundle](../2026-10-26-br4-probabilistic-drafting/README.md)):

| card | arm | speculative config | block | tool turns emitted, pass 1 / pass 2 | text-turn emissions |
|---|---|---|---|---|---|
| BR4 | A | v26 (greedy drafting) | – | 6 / 7 | 4 / 3 |
| BR4 | P | + probabilistic | – | 6 / 8 | 5 / 4 |
| BR4 | PB | + probabilistic + block | ✓ | **8 / 9** | 6 / 4 |
| BR4 | A2 | v26 (greedy drafting) | – | 8 / 8 | 4 / 4 |
| BR5 | A | v28-prob | – | 9 / 8 | 4 / 3 |
| BR5 | B | v28-prob + block | ✓ | **6 / 7** | 6 / 4 |
| BR5 | A2 | v28-prob | – | 8 / 8 | 5 / 4 |

Across the five probabilistic-drafting arms (10 passes) the counts are 6, 6, 7, 8, 8, 8, 8, 8, 9, 9; across all seven
arms (14 passes) 6 ×3, 7 ×2, 8 ×7, 9 ×2. The two block-on arms sit at **both** ends of that spread, and the block-off
arms cover 6–9 on their own. The count does not follow the block key. Greedy rows never use block verification, so
at temperature 0 B must equal A by construction; what moves the count is run-to-run nondeterminism in batched MoE plus
speculative decode, which is not bit-deterministic on this lane. **A range drawn from two controls × two passes is
too narrow a bar for that replay:** in BR4 the two controls alone spanned 6–8, in BR5 8–9, and a candidate with the
controls' own distribution trips it often.

That reading is consistent with every receipt but is not itself a tested result, and the verdict stands as
registered.

### Decision

Per the pre-registration: **PB (v28-prob + block verification) is RETIRED on this lane** pending **Card BR6**, which
repeats this bracket with a redesigned G6 (queued 2026-10-26; not yet run):

- **4 replay passes per arm** instead of 2;
- **a pooled-control bar ±1** (all control passes pooled, widened by one tool turn) instead of the bare two-control
  range;
- **a control-agreement validity check**, so a card whose two controls disagree with each other is flagged before the
  candidate is scored against them;
- **a per-transcript annotation** (which of the 12 turns emitted, per pass), so a systematic change on one transcript
  is visible separately from the pass count.

The exact BR6 bars are set in its own pre-registration, before launch.

The reference stays **v28-prob** (unchanged; `dsv41-vllm-1017-v28-prob-BOUND-REF` stopped-and-kept, v26seed is its
rollback). `:30006` keeps serving the sidecar sibling's v27 pair.

The speed and fidelity story for block-on-probabilistic is consistent across three cards: **+2.0% tokens per verify
step** on the sidecar lane (Card BR1,
[bundle](../../../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/README.md)),
**+1.5%** on top of P in BR4 ([bundle](../2026-10-26-br4-probabilistic-drafting/README.md)), and **+2.5%** here, with
TF flips at the control floor every time. With greedy drafting the same key is a null (Card BR3,
[bundle](../2026-10-26-br3-block-verification-null/README.md)). Write-up:
[al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html](https://al-engr.com/gb300-vs-4x-rtx6000-dsv41-flash-brrr.html).

## Files

| path | what it is |
|---|---|
| `PREREG-BR5.md` | the pre-registration, written before launch (11:33:05 CDT) |
| `VERDICT.txt`, `verdict.json` | the pre-registered scoring (`scripts/br5_verdict.py`), including the BR4 parity re-check annotation |
| `SUMMARY.txt`, `runner.log`, `phases.jsonl` | runner timeline: dark, isolate, boots, facts, GO/unfreeze, cells, restore |
| `instruments.txt` | sha1 of every script and fixture as run **on the box** (the published copies differ by the path scrub only), image digests, A/B `SPEC_EXTRA`, GO ctrl, sampling |
| `launch-*.txt`, `health-*.json`, `errcount-*.txt`, `oom-*.txt`, `mem.tsv`, `settle-pre-*.txt`, `preflight-*.txt`, `kv-ref.txt` | launcher output, chat health, error-line counts, OOM counts, MemAvailable samples, pre-boot host state, the v28-prob KV reference |
| `raw/<arm>/` | `facts.json`, `boot-facts.log`, `boot-autotune.log`, `boot-block.log`, `boot-prob.log`, `boot-pin.log` (hook rehome lines), `args.json` / `env.json` / `binds.json` (from `docker inspect`), `go.json` + `go.log` + `ar-ctrl.json` + `ar-lines.tsv` (adaptive GO/unfreeze and swap/telemetry line counts per cell), `perclass-t1/` and `perclass-t0/` (tables md+json; **per-request `rows` stripped**, `rows_withheld` set), `mixed-*.json.gz` (per-request timings only; synthetic prompt pools), `specmetrics/` (`vllm:spec_decode*` counters per window, including the per-position counters), `turns-*-summary.json` |
| `raw/tf-compare-*.json`, `raw/tfcmp-*.log`, `raw/flipsm-*.txt` | teacher-forced compares and margin-weighted flips |
| `raw/gauntlet-<arm>/` | gauntlet summary + sandbox isolation check (fixture inputs and per-task jsonl omitted) |
| `raw/production/` | production lane (sidecar v27) args/env before and after the card, plus the v28-prob reference args/env/binds that G0 compared against |
| `raw/restore-v27-block.facts.log` | filtered KV / autotune / startup lines from the production restore boot |
| `scripts/` | `cardBR5_runner.sh`, `br5_verdict.py`, `DIFF-br5-runner-vs-br3.txt`, `DIFF-br5-verdict-vs-br3.txt`, plus `launch_br3.sh` and `DIFF-br3-launch.txt` reused from BR3 (same sha1 `8066045c`), and the BR1 instruments it reuses unchanged (`perclass_run_t.sh`, `perclass_decode_t.py`, `ar_mixed_t.py` and their `DIFF-*.txt`; same sha1 on the box as in the BR1/BR3/BR4 bundles), exactly as run, with box paths scrubbed |
| `throughput.csv` | every speed / acceptance / per-position / KV / flip cell with its receipt |

## Withheld, on purpose

- The per-class prompt file `perclass_prompts.jsonl` (sha1 `618898aac432`) is real Hermes transcripts and stays
  **private**. The tables are published verbatim; the per-request `rows` (output heads) are removed from every
  perclass/agentfix JSON. The `turns-*.json` replays (from `transcript_fixture.json`, also private transcripts) are
  replaced by per-pass counts. The teacher-forced per-position dumps (`tf-*.jsonl`) stay on the box; the compares
  are published.
- Server post-cell logs (`postlog-*`), full boot logs, the hook's `swaps.jsonl` / `telemetry.jsonl` / `steps.bin`,
  `preflight-docker-ps-a.txt`, `verdict.err`, the gauntlet per-task jsonl and fixture trees, and the full restore log
  stay in the box receipts.
- The desk brief and dry-run fabricator (`BRIEF-BR5.md`, `dryrun/`) stay on the box; the dry-run numbers are
  synthetic by design.
- Path scrub: the box home directory → `~`, LAN address → `<gb300>`.
