# 2026-10-26 — Card BR4: probabilistic drafting on the GB300-only v26 lane (PASS → v28-prob); + block = candidate

Two candidates were bracketed against the v26 "Seeded Adaptive" speculative config:

- **P** = v26 + `"draft_sample_method":"probabilistic"` (the drafter samples its tokens instead of taking the argmax);
- **PB** = P + `"rejection_sample_method":"block"` (block verification, Sun et al.).

Everything else is the v26 lane exactly as `dsv41-vllm-1017-v26seed-BOUND-REF` booted: `launch_br3.sh` (= the v26
`launch_ar6.sh` + `SPEC_EXTRA`, reused unchanged from Card BR3, sha1 `8066045c`), image
`vllm/vllm-openai:deepseekv41-flash-0909` (`sha256:00d577a6…`), hook v26ar.2, seed map `rowmap-mixed-ar-v1.json`,
UVA offload 60 GiB, 1M context, 24 seqs, DSpark k-schedule `[[1,8,5],[9,24,1]]`. The control arms A/A2 draft greedy
(no `draft_sample_method` key) and render byte-identical to the v26seed Args (G0). The launch delta is the BR3 one,
[`scripts/DIFF-br3-launch.txt`](scripts/DIFF-br3-launch.txt); the arm's `SPEC_EXTRA` is in
[`instruments.txt`](instruments.txt).

Bracketed **A (control) → P → PB → A2 (control)**, 2026-10-26 07:23–10:28 CDT. Each arm was one fresh server boot
(this lane has no sidecar). The hook booted frozen on every arm, then got `GO` + `ctrl.json = {"freeze": false,
"baseline": true}` and 90 s of settle, so adaptive residency ran unfrozen on all four arms, as in production
(`raw/<arm>/go.json`, `raw/<arm>/ar-lines.tsv`). The production lane (the sidecar recipe's v27 pair) was dark for the
card and firewalled to loopback; 0 foreign requests reached any arm. It was restored at 10:28 with its fail-stop
watchdog re-armed (PPID 1).

## Why the primary window is at temperature 1.0

Same instrument as Card BR3 ([`../2026-10-26-br3-block-verification-null/`](../2026-10-26-br3-block-verification-null/README.md))
and the sidecar recipe's Card BR1
([`../../../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/`](../../../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/README.md)).
Both keys act only on non-greedy rows: at temperature 0 the drafter's sample is its argmax and the rejection kernel
takes the greedy branch, so P and PB must equal A there. The primary speed window is `perclass_run_t.sh` +
`perclass_decode_t.py` at **temperature 1.0, top_p 1.0, seeded** (seed 42 + row index): 40 real Hermes-transcript
prompts per class, 256 forced tokens, r1/r2 at C1 and C8. The stock temperature-0 window ran on every arm as a
**parity** gate (G2b: the candidate must equal A within noise).

All four arms were autotune **hash hits** (Loaded 189 / Saved 189 configs, same cache dir, 5–6 s autotune span).
Boots took 474 / 485 / 484 / 484 s (A / P / PB / A2).

## Verdict

[`VERDICT.txt`](VERDICT.txt), [`verdict.json`](verdict.json): **`PASS-P`, suffix `BR4-PPASS_PBFAIL-G2b-G6`**,
pre-registered, not re-scored.

- **P passes every gate** (G0 binding, G0b foreign, G1 controls, G2 per-class temp 1.0, G2b temp-0 parity, G3 mixed,
  G4 teacher-forced, G5 gauntlet, G6 turns, G7 errors).
- **PB fails two parity gates**, G2b and G6 (below). All its speed and fidelity gates pass.

## Results

| temp 1.0 | A | **P** | PB | A2 | P ÷ ctrl | PB ÷ ctrl | PB ÷ P |
|---|--:|--:|--:|--:|--:|--:|--:|
| accepted tokens per verify step, per-class window | 3.122 | **3.438** | 3.491 | 3.115 | **1.103** | 1.120 | 1.015 |
| accepted tokens per verify step, mixed C1 | 2.087 | **2.100** | 2.124 | 2.087 | 1.006 | 1.018 | 1.012 |
| mixed C1 tok/s (`ar_mixed_t.py`, 3 × 120 s) | 179.79 | **181.09** | 182.45 | 179.93 | 1.007 (+0.68%) | 1.014 | 1.008 |

ctrl = mean(A, A2). Temperature-1.0 per-class decode tok/s (mean of r1/r2):

| class | C1 A / A2 / **P** / PB | C1 P ÷ ctrl | C1 PB ÷ ctrl | C8 A / A2 / **P** / PB | C8 P ÷ ctrl | C8 PB ÷ ctrl |
|---|---|--:|--:|---|--:|--:|
| prose | 197.4 / 197.2 / **217.2** / 219.9 | **1.101** | 1.114 | 451.8 / 454.6 / **494.3** / 510.1 | **1.091** | 1.126 |
| structured | 219.0 / 217.2 / **241.9** / 248.4 | **1.109** | 1.139 | 539.7 / 550.0 / **589.6** / 611.4 | **1.082** | 1.122 |
| code | 255.4 / 260.9 / **269.9** / 278.3 | **1.045** | 1.078 | 580.9 / 559.6 / **626.8** / 644.6 | **1.099** | 1.130 |
| shell_ops | 291.9 / 282.6 / **319.8** / 323.4 | **1.113** | 1.126 | 642.2 / 635.8 / **723.3** / 739.1 | **1.132** | 1.157 |
| tool_json | 254.6 / 251.4 / **294.0** / 292.0 | **1.162** | 1.154 | 671.9 / 661.9 / **791.8** / 780.0 | **1.187** | 1.170 |

Per-class accepted tokens per verify step (P / PB / ctrl): prose C1 2.81 / 2.89 / 2.56, C8 2.89 / 2.94 / 2.57;
structured 3.05 / 3.16 / 2.77 and 3.07 / 3.09 / 2.77; code 3.73 / 3.85 / 3.56 and 3.85 / 3.88 / 3.57; shell_ops
4.10 / 4.10 / 3.69 and 4.03 / 4.01 / 3.60; tool_json 3.71 / 3.73 / 3.23 and 3.67 / 3.72 / 3.26.

**Per-position acceptance** (temp-1.0 window, accepted at draft position *i* ÷ drafts, from
`vllm:spec_decode_num_accepted_tokens_per_pos`):

| position | 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|
| A | 0.699 | 0.502 | 0.378 | 0.299 | 0.245 |
| **P** | **0.785** | **0.587** | **0.443** | **0.346** | **0.278** |
| PB | 0.786 | 0.598 | 0.457 | 0.360 | 0.290 |
| A2 | 0.699 | 0.500 | 0.376 | 0.296 | 0.244 |

P lifts acceptance at every position by 0.03–0.09 over mean(A, A2) (0.086 at position 1, 0.034 at position 5). PB adds 0.001–0.014 on top.

- **G1 controls:** A vs A2 agree at C1 within 0.1–3.2% per class at temp 1.0, and 0.1–2.0% at temp 0.
- **G4 teacher-forced** (79,544 positions, 39 docs, 0 token-mismatch docs): top-1 flips P-vs-A 1.129%, P-vs-A2
  1.098%, PB-vs-A 1.095%, PB-vs-A2 1.099%, against an A2-vs-A floor of 1.136% (bar ≤ 1.5 × floor). Margin-weighted
  flips at ref p ≥ 0.9 on agent/tool docs: 0 on every compare ([`raw/flipsm-*.txt`](raw/)). P-vs-PB 1.032%.
  This is a prefill-only check of the target model; the drafter and verifier do not run in it, so it shows the
  served model is the same on every arm.
- **G5 gauntlet** 12/12 on all four arms. **G6 replay_turns** tool-turn emission per pass: P [6, 8]/12 against a
  control range of 6–8 (A [6, 7], A2 [8, 8]).
- **G7 errors:** 0 FF errors and 0 foreign requests on all arms.
- **x3 acceptance canary** (temp 0, inside each window): identical across A/P/PB/A2 on every class.
- **KV:** 2,495,694 tokens on A and A2 (= the v26seed boot log) and **2,481,694 on P and PB (−0.56%)**. The brief
  predicted this before the run: with probabilistic drafting vLLM allocates a bf16 draft-logits cache
  (24 seqs × 5 × 129,280 vocab × 2 B ≈ 31 MB) in `init_speculator`, before KV profiling, and
  14,000 tokens × ~2,254 B/token ≈ 31.6 MB. G0 allowed a candidate drop of up to 1.5%. Max concurrency at 1M tokens
  reads 2.37× instead of 2.38×.
- **Adaptive residency was live during the cells.** Swaps between pct1-start and turns-end: A 86,566, P 83,214,
  PB 82,571, A2 86,392.

### Why PB failed (and why it stays a candidate)

Both PB failures are on **temperature-0 parity gates**, where neither key does anything to a greedy row:

- **G2b shell_ops C1 +3.6%** against a 2.6% bar (the bar is 1.5 × the control's own r1/r2 spread of 1.7%). The other
  four classes sat inside their bars (prose +1.1%, structured −0.9%, code −0.3%, tool_json +3.9% vs a 13.1% bar).
- **G6 replay_turns** emitted tool calls on 9/12 tool turns in one pass ([8, 9]) against a control range of 6–8.

Both trips are in the "faster / more" direction. We read them as run-to-run nondeterminism (batched MoE plus
speculative decode are not bit-deterministic) against two tight bars, not as a regression. That reading is not
tested by a receipt, and the pre-registered verdict stands. PB's temperature-1.0 numbers are the best of the four
arms (+12.0% tokens per step, PB ÷ P 1.015), so it stays a **candidate**, to be re-bracketed alone against v28-prob.

### Mechanism

**Our reading.** At temperature > 0 the target samples. A greedy drafter always proposes its argmax; treated as a
one-hot draft distribution, that token survives the p/q ratio test only with the target's probability for it. A
drafter that samples from its own distribution proposes tokens shaped like the target's sample, and the full ratio
test min(1, p_target / q_draft) accepts more of them. Acceptance rose at every
draft position (+0.03–0.09), which is consistent with that.

This also explains Card BR3. Block verification reweights the draft distribution against the target's, so it needs
a real draft distribution. With greedy drafting it had nothing to work on (BR3: 0.999× tokens per step). With
probabilistic drafting it adds +1.5% on top (PB ÷ P), the same order as Card BR1 on the sidecar lane (+2.0%), which
drafts probabilistically. Greedy drafting was this lane's missing lever.

**Decision:** **P is promoted as this recipe's reference, `v28-prob`.** The container
`dsv41-vllm-1017-v28-prob-BOUND-REF` was booted once by [`scripts/park_v28_prob.sh`](scripts/park_v28_prob.sh)
with the production soak `/ar` bind, verified, and stopped-and-kept; its log is
[`park-v28.log`](park-v28.log). It bound at 10:57:41 CDT after 476 s with `GPU KV cache size: 2,481,694 tokens`,
`draft_sample_method` in Args and `rejection_sample_method` absent, rehome done, and passed a chat smoke
(`'v28 ok'`) and a tool-call smoke (`['shell']`): park facts verdict `OK`. The lane was dark 10:49–11:06 CDT; the
v27 pair rebound (KV 4,400,013) and its fail-stop watchdog was re-armed with PPID 1 (`===== PARK v28-prob END`). `dsv41-vllm-1017-v26seed-BOUND-REF` is kept as the rollback. **`:30006` keeps serving
the sidecar sibling's v27 pair**: this is a reference update, not a routing change. PB stays a candidate.

## Files

| path | what it is |
|---|---|
| `VERDICT.txt`, `verdict.json` | the pre-registered scoring (`scripts/br4_verdict.py`) |
| `SUMMARY.txt`, `runner.log`, `phases.jsonl` | runner timeline: dark, isolate, boots, facts, GO/unfreeze, cells, restore |
| `instruments.txt` | sha1 of every script and fixture as run **on the box** (the published copies differ by the path scrub only), image digests, P/PB `SPEC_EXTRA`, GO ctrl, sampling |
| `launch-*.txt`, `health-*.json`, `errcount-*.txt`, `oom-*.txt`, `mem.tsv`, `settle-pre-*.txt`, `preflight-*.txt`, `kv-ref.txt` | launcher output, chat health, error-line counts, OOM counts, MemAvailable samples, pre-boot host state, the v26seed KV reference |
| `raw/<arm>/` | `facts.json`, `boot-facts.log`, `boot-autotune.log`, `boot-block.log`, `boot-prob.log` (new in BR4: draft-sampling boot lines), `boot-pin.log` (hook rehome lines), `args.json` / `env.json` / `binds.json` (from `docker inspect`), `go.json` + `go.log` + `ar-ctrl.json` + `ar-lines.tsv` (adaptive GO/unfreeze and swap/telemetry line counts per cell), `perclass-t1/` and `perclass-t0/` (tables md+json; **per-request `rows` stripped**, `rows_withheld` set), `mixed-*.json.gz` (per-request timings only; synthetic prompt pools), `specmetrics/` (`vllm:spec_decode*` counters per window, including the per-position counters), `turns-*-summary.json` |
| `raw/tf-compare-*.json`, `raw/tfcmp-*.log`, `raw/flipsm-*.txt` | teacher-forced compares and margin-weighted flips |
| `raw/gauntlet-<arm>/` | gauntlet summary + sandbox isolation check (fixture inputs and per-task jsonl omitted) |
| `raw/production/` | production lane (sidecar v27) args/env before and after the card, plus the v26seed reference args/env/binds that G0 compared against |
| `raw/restore-v27-block.facts.log` | filtered KV / autotune / startup lines from the production restore boot |
| `park-v28.log` | the v28-prob reference park run (boot once, facts, smoke, stop-and-keep, restore v27) |
| `scripts/` | `cardBR4_runner.sh`, `br4_verdict.py`, `DIFF-br4-runner-vs-br3.txt`, `DIFF-br4-verdict-vs-br3.txt`, `park_v28_prob.sh`, plus `launch_br3.sh` and `DIFF-br3-launch.txt` reused from BR3 (same sha1 `8066045c`), and the BR1 instruments it reuses unchanged (`perclass_run_t.sh`, `perclass_decode_t.py`, `ar_mixed_t.py` and their `DIFF-*.txt`; same sha1 on the box as in the BR1/BR3 bundles), exactly as run, with box paths scrubbed |
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
- The desk brief and dry-run fabricator (`BRIEF-BR4.md`, `dryrun/`) stay on the box; the dry-run numbers are
  synthetic by design.
- Path scrub: the box home directory → `~`, LAN address → `<gb300>`.
