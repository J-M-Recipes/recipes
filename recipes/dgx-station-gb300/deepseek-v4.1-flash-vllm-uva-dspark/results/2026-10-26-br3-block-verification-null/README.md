# 2026-10-26 — Card BR3: DSpark block verification on the GB300-only v26 lane (a NULL)

One key was added to the v26 "Seeded Adaptive" speculative config: `"rejection_sample_method":"block"` (block
verification, Sun et al.). Everything else is the v26 lane exactly as `dsv41-vllm-1017-v26seed-BOUND-REF` booted:
`launch_ar6.sh` (here `launch_br3.sh` = `launch_ar6.sh` + `SPEC_EXTRA`), image `vllm/vllm-openai:deepseekv41-flash-0909`
(`sha256:00d577a6…`), hook v26ar.2, seed map `rowmap-mixed-ar-v1.json`, UVA offload 60 GiB, 1M context, 24 seqs,
DSpark k-schedule `[[1,8,5],[9,24,1]]`, **greedy drafting** (no `draft_sample_method` in the config). The launch delta
is in [`scripts/DIFF-br3-launch.txt`](scripts/DIFF-br3-launch.txt): three hunks, and arm A/A2 render byte-identical
to the v26seed Args.

Bracketed **A (control) → B (block) → A2 (control)**, 2026-10-26 04:26–06:49 CDT. Each arm was one fresh server
boot (this lane has no sidecar). The hook booted frozen on every arm, then got `GO` + `ctrl.json = {"freeze": false,
"baseline": true}` and 90 s of settle, so adaptive residency ran unfrozen on all three arms, as in production
(`raw/<arm>/go.json`, `raw/<arm>/ar-lines.tsv`). The production lane (the sidecar recipe's v27 pair) was dark for the
card and firewalled to loopback; 0 foreign requests reached any arm. It was restored at 06:49 with its fail-stop
watchdog re-armed (PPID 1).

## Why the primary window is at temperature 1.0

Same instrument as the sidecar recipe's Card BR1
([`../../../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/`](../../../deepseek-v4.1-flash-vllm-uva-dspark-sidecar/results/2026-10-26-br1-block-verification/README.md)):
vLLM applies block verification only on the non-greedy path. At temperature 0 the key is a no-op. The primary speed
window is `perclass_run_t.sh` + `perclass_decode_t.py` at **temperature 1.0, top_p 1.0, seeded** (seed 42 + row
index): 40 real Hermes-transcript prompts per class, 256 forced tokens, r1/r2 at C1 and C8. The stock temperature-0
window ran on every arm as a sanity gate (G2b: B must equal A within noise).

All three arms were autotune **hash hits** (Loaded 189 / Saved 189 configs, same cache dir, 5 s autotune span).
Boots took 474 / 484 / 484 s. KV was 2,495,694 tokens on every arm, equal to the v26seed boot log.

## Verdict

[`VERDICT.txt`](VERDICT.txt), [`verdict.json`](verdict.json): **FAIL `BR3FAIL-G2`**, pre-registered. There was no
re-score. Every other gate passed (G0 binding, G0b foreign, G1 controls, G2b temp-0 no-op, G3 mixed, G4 teacher-forced,
G5 gauntlet, G6 turns, G7 errors).

G2 failed on **structured**: C1 0.973 (bar −2.1%) and C8 0.970 (bar −2.0%).

## Results

**The headline is a null.** Block verification does not move acceptance on this lane.

| | A | **B** | A2 | B ÷ mean(A, A2) |
|---|--:|--:|--:|--:|
| tokens per verify step, temp-1.0 per-class window | 3.111 | **3.116** | 3.125 | **0.999** |
| tokens per verify step, temp-1.0 mixed C1 | 2.087 | **2.093** | 2.087 | 1.003 |
| mixed C1 tok/s (`ar_mixed_t.py`, 3 × 120 s, temp 1.0) | 180.07 | **179.54** | 179.87 | 0.998 (−0.24%; bar −2%) |

Temperature-1.0 per-class decode tok/s (mean of r1/r2), B ÷ mean(A, A2):

| class | C1 A / A2 / **B** | C1 ratio | C8 A / A2 / **B** | C8 ratio | controls' own C8 spread (A vs A2) |
|---|---|--:|---|--:|--:|
| prose | 200.7 / 202.1 / **201.4** | 1.000 | 446.7 / 469.4 / **453.0** | 0.989 | 5.0% |
| structured | 214.7 / 217.8 / **210.3** | **0.973** ✗ | 543.2 / 539.6 / **525.2** | **0.970** ✗ | 0.7% |
| code | 255.1 / 260.2 / **253.5** | 0.984 | 522.4 / 545.0 / **580.7** | 1.088 | 4.2% |
| shell_ops | 287.1 / 285.5 / **285.0** | 0.995 | 640.2 / 650.9 / **674.2** | 1.044 | 1.7% |
| tool_json | 255.5 / 253.4 / **258.2** | 1.015 | 641.0 / 701.8 / **729.3** | 1.086 | 9.1% |

- The C8 code and tool_json rises sit inside what the two controls disagree by at C8 (4.2% and 9.1%), so they are not
  read as gains. The per-class acceptance tells the same story: tokens/step B vs controls is code 3.47/3.56 at C1 and
  3.57/3.43 at C8, tool_json 3.31/3.21 and 3.40/3.28, structured 2.67/2.73 and 2.70/2.75. Those moves are small,
  run in both directions, and net to 0.999 over the window.
- **G1 controls:** A vs A2 agree at C1 within 0.6–2.0% per class at temp 1.0, and 0.2–4.2% at temp 0.
- **G2b temp-0 no-op:** B is within noise on every class: prose −3.5%, structured −0.4%, code −0.4%, shell_ops −0.9%,
  tool_json −0.0%.
- **G4 teacher-forced** (79,544 positions, 39 docs, 0 token-mismatch docs): top-1 flips B-vs-A 1.086% and B-vs-A2
  1.128%, against an A2-vs-A floor of 1.092%. Margin-weighted flips at ref p ≥ 0.9 on agent/tool docs: 0 on all three
  compares ([`raw/flipsm-*.txt`](raw/)).
- **G5 gauntlet** 12/12 on A, B and A2. **G6 replay_turns** tool-turn emission B [8, 7]/12, against A [8, 7] and
  A2 [6, 8].
- **G7 errors:** 0 FF errors and 0 foreign requests on all arms.
- **x3 acceptance canary** (temp 0, inside each window): identical across A/B/A2 on every class.
- **Adaptive residency was live during the cells.** Swaps between pct1-start and turns-end: A 86,282, B 87,151,
  A2 86,330.

**Read (our interpretation; no receipt tests it):** this lane drafts greedy. Block verification reweights the draft
distribution against the target's, but a greedy draft is one-hot, which leaves it nothing to reweight. It then
degenerates to standard verification, and the 0.999 tokens/step is what that looks like. The sidecar lane drafts
probabilistically (`draft_sample_method: probabilistic`). On that lane the same key gained +2.0% tokens/step and was
promoted as v27 (Card BR1). The one gate that failed, structured C1/C8 at −3%, is a speed cell on a config that did
not change acceptance. We do not claim block verification slows structured text.

**Decision:** **NOT promoted** on this lane. Next is **Card BR4**: probabilistic drafting alone, then
probabilistic + block, in the same A → B → A2 bracket (queued 2026-10-26).

## Files

| path | what it is |
|---|---|
| `VERDICT.txt`, `verdict.json` | the pre-registered scoring (`scripts/br3_verdict.py`) |
| `SUMMARY.txt`, `runner.log`, `phases.jsonl` | runner timeline: dark, isolate, boots, facts, GO/unfreeze, cells, restore |
| `instruments.txt` | sha1 of every script and fixture as run **on the box** (the published copies differ by the path scrub only), image digests, B's SPEC_EXTRA, GO ctrl, sampling |
| `launch-*.txt`, `health-*.json`, `errcount-*.txt`, `oom-*.txt`, `mem.tsv`, `settle-pre-*.txt`, `preflight-*.txt`, `kv-ref.txt` | launcher output, chat health, error-line counts, OOM counts, MemAvailable samples, pre-boot host state, the v26seed KV reference |
| `raw/<arm>/` | `facts.json`, `boot-facts.log`, `boot-autotune.log`, `boot-block.log`, `boot-pin.log` (hook rehome lines), `args.json` / `env.json` / `binds.json` (from `docker inspect`), `go.json` + `go.log` + `ar-ctrl.json` + `ar-lines.tsv` (adaptive GO/unfreeze and swap/telemetry line counts per cell), `perclass-t1/` and `perclass-t0/` (tables md+json; **per-request `rows` stripped**, `rows_withheld` set), `mixed-*.json.gz` (per-request timings only; synthetic prompt pools), `specmetrics/` (`vllm:spec_decode*` counters per window), `turns-*-summary.json` |
| `raw/tf-compare-*.json`, `raw/tfcmp-*.log`, `raw/flipsm-*.txt` | teacher-forced compares and margin-weighted flips |
| `raw/gauntlet-<arm>/` | gauntlet summary + sandbox isolation check (fixture inputs and per-task jsonl omitted) |
| `raw/production/` | production lane (sidecar v27) args/env before and after the card, plus the v26seed reference args/env/binds that G0 compared against |
| `raw/restore-v27-block.facts.log` | filtered KV / autotune / startup lines from the production restore boot |
| `scripts/` | `cardBR3_runner.sh`, `launch_br3.sh`, `br3_verdict.py`, `DIFF-br3-launch.txt`, `DIFF-br3-runner-vs-br1.txt`, `DIFF-br3-verdict-vs-br1.txt`, plus the BR1 instruments it reuses unchanged (`perclass_run_t.sh`, `perclass_decode_t.py`, `ar_mixed_t.py` and their `DIFF-*.txt`; same sha1 on the box as in the BR1 bundle), exactly as run, with box paths scrubbed |
| `throughput.csv` | every speed / acceptance / flip cell with its receipt |

## Withheld, on purpose

- The per-class prompt file `perclass_prompts.jsonl` (sha1 `618898aac432`) is real Hermes transcripts and stays
  **private**. The tables are published verbatim; the per-request `rows` (output heads) are removed from every
  perclass/agentfix JSON. The `turns-*.json` replays (from `transcript_fixture.json`, also private transcripts) are
  replaced by per-pass counts. The teacher-forced per-position dumps (`tf-*.jsonl`) stay on the box; the compares
  are published.
- Server post-cell logs (`postlog-*`), full boot logs, the hook's `swaps.jsonl` / `telemetry.jsonl` / `steps.bin`,
  `preflight-docker-ps-a.txt`, the gauntlet per-task jsonl and fixture trees, and the full restore log stay in the
  box receipts.
- The desk brief and dry-run fabricator (`BRIEF-BR3.md`, `dryrun/`) stay on the box; the dry-run numbers are
  synthetic by design.
- Path scrub: the box home directory → `~`, LAN address → `<gb300>`.
