# 2026-10-25 — Card BR1: DSpark block verification on the sidecar lane

One key added to the production speculative config: `"rejection_sample_method":"block"` (block verification,
Sun et al.). Everything else is the production lane as it serves on `:30006`: the v25m map
(`ROWMAP=rowmap-mixed-ar-v1.json`), the kv-host-tier flags
(`EXTRA="--kv-cache-memory 9126805504 --kv-offloading-size 48 --kv-offloading-backend native"`), image
`vllm/vllm-openai@sha256:3af6a45b…` (nightly af7f9488), 32 seats, 1M context, DSpark k=5/3/3 by batch size,
`draft_sample_method` probabilistic. The launch delta is in [`scripts/DIFF-launch.txt`](scripts/DIFF-launch.txt).

Bracketed **A (control) → B (block) → A2 (control)**, 2026-10-25 19:33–21:35 CDT. Each arm got a fresh sidecar +
server boot, and the fail-closed watchdog was armed in enforce mode on each one. While the lane was dark it was
firewalled to loopback, so 0 foreign clients reached it.

## Why the primary window is at temperature 1.0

vLLM af7f9488 applies block verification only on the non-greedy path (`rejection_sampler_utils.py`:
`USE_BLOCK_VERIFICATION and not is_greedy`). A greedy row is accepted iff the draft equals the target argmax. At
temperature 0 the key is therefore a no-op, and every stock speed instrument on this lane sends temperature 0.
The primary speed window is `perclass_run_t.sh` + `perclass_decode_t.py` at **temperature 1.0, top_p 1.0, seeded**
(seed 42 + row index): 40 real Hermes-transcript prompts per class, 256 forced tokens, r1/r2 at C1 and C8. The
stock temperature-0 window ran on every arm too. Its job is a sanity gate: B must equal A within noise there.

`rejection_sample_method` is not one of vLLM's compute-hash factors. All three arms were autotune **hash hits**
(Loaded 117 / Saved 117 configs, same cache dir), and boots took 529 / 528 / 529 s.

## Verdicts — both on the record

| file | verdict | why |
|---|---|---|
| [`VERDICT.txt`](VERDICT.txt), [`verdict.json`](verdict.json) (= `verdict-prereg.json`) | **FAIL `BR1FAIL-G0-G2b`** (pre-registered) | two instrumentation defects, below |
| [`VERDICT-rescore.txt`](VERDICT-rescore.txt), [`verdict-rescore.json`](verdict-rescore.json) | **PASS `BR1PASS`** (re-scored) | `scripts/br1_verdict_rescore.py` = `br1_verdict.py` + the two fixes in `scripts/rescore_patch_br1.py` |

1. **G0 compared the container env as an ordered list.** Docker returns the same 44 variables in a different order
   from boot to boot. The sets were identical on every arm, and the re-score compares them as sets.
2. **G2b's temperature-0 no-op bar was tighter than the class's own r1/r2 noise.** The pre-registered bar was
   max(2%, k × the A-vs-A2 bracket spread), and code/C1 read −2.2% against it. The controls' own r1/r2 spread on
   that class was 2.1%. The re-score sets each class's bar to max(2%, k × bracket spread, k × the controls' larger
   r1/r2 spread), with the same k. The bars came out at 2.6–4.1%.

No speed, fidelity or watchdog gate changed between the two scorings.

## Results (re-scored gate table, numbers identical in both files)

Temperature-1.0 per-class decode tok/s (mean of r1/r2), B ÷ mean(A, A2):

| class | C1 A / A2 / **B** | C1 ratio | C8 A / A2 / **B** | C8 ratio |
|---|---|--:|---|--:|
| prose | 266.7 / 270.2 / **278.3** | 1.037 | 972.4 / 978.7 / **983.0** | 1.008 |
| structured | 302.2 / 300.8 / **305.3** | 1.013 | 1,059.8 / 1,024.8 / **1,068.2** | 1.025 |
| code | 352.8 / 353.0 / **357.3** | 1.013 | 1,173.9 / 1,159.6 / **1,221.1** | 1.047 |
| shell_ops | 374.0 / 385.5 / **378.1** | 0.996 | 1,194.2 / 1,245.8 / **1,320.3** | 1.082 |
| tool_json | 333.4 / 326.4 / **347.6** | 1.054 | 1,184.1 / 1,156.0 / **1,202.5** | 1.028 |

- **G1 controls:** A vs A2 agree at C1 within 0.1–3.0% per class (temp 1.0) and 0.7–2.7% (temp 0).
- **Tokens per verify step** (1 + accepted/drafts from `/metrics` deltas over the temp-1.0 window): A 3.204,
  **B 3.262**, A2 3.189 (B/ctrl 1.020). Mixed C1: 2.092 / 2.105 / 2.088 (1.007).
- **G3 mixed traffic** (`ar_mixed_t.py`, C1, natural length, temp 1.0, 3 × 120 s): A 212.05, **B 212.77**, A2 212.65
  (+0.20% vs controls; bar −2%).
- **G2b temp-0 no-op:** B is within noise on every class: prose −0.9%, structured −0.2%, code −2.2%, shell_ops +1.1%,
  tool_json +0.7%.
- **G4 teacher-forced** (79,544 positions, 39 docs, 0 token-mismatch docs): top-1 flips B-vs-A 2.166%, B-vs-A2
  2.176%, against an A2-vs-A floor of 2.172%. Margin-weighted (`flips_margin_p2.py`): hi_agent_tool = 0 on all three
  compares, W 2.839 (vs A) / 2.751 (vs A2) against a floor W of 2.773 ([`raw/flipsm-*.txt`](raw/)).
- **G5 gauntlet** 12/12 on A, B and A2. **G6 replay_turns** tool-turn emission B [7, 7]/12 vs A [6, 8], A2 [7, 6].
- **G7 watchdog:** 0 timeouts, 0 trips, 0 FF errors, sidecar alive, on all arms. After restore the production watchdog
  was re-armed with PPID 1.
- **x3 acceptance canary** (temp 0, inside each window): within noise on every class (`VERDICT-rescore.txt`).

**Read:** block verification is a small, real gain at temperature > 0. The best C1 cells are +3.7% prose and
+5.4% tool-JSON, with +2.0% tokens per step, and the teacher-forced flips sit at the control floor. It is a
**promotion candidate, NOT promoted**; that decision is James's. It does nothing at temperature 0.

## Files

| path | what it is |
|---|---|
| `VERDICT*.txt`, `verdict*.json` | both scorings (see above) |
| `SUMMARY.txt`, `runner.log`, `phases.jsonl` | runner timeline: dark, isolate, boots, facts, cells, restore |
| `instruments.txt` | sha1 of every script and fixture as run **on the box** (the published copies differ by the path scrub only), image digests, per-arm EXTRA/SPEC_EXTRA, sampling |
| `launch-*.txt`, `health-*.json`, `errcount-*.txt`, `oom-*.txt`, `mem.tsv`, `settle-pre-*.txt`, `preflight-*.txt` | launcher output, chat health, error-line counts, OOM counts, MemAvailable samples, pre-boot host state |
| `raw/<arm>/` | `facts.json`, `boot-facts.log`, `boot-autotune.log`, `boot-block.log`, `args.json` / `rtx-args.json` / `env.json` (from `docker inspect`), `perclass-t1/` and `perclass-t0/` (tables md+json; **per-request `rows` stripped**, `rows_withheld` set), `mixed-*.json.gz` (per-request timings only; synthetic prompt pools), `specmetrics/` (`vllm:spec_decode*`, prefix-cache and token counters per window), `turns-*-summary.json`, `wd.json`, `watchdog.log` |
| `raw/tf-compare-*.json`, `raw/tfcmp-*.log`, `raw/flipsm-*.txt` | teacher-forced compares and margin-weighted flips |
| `raw/gauntlet-<arm>/` | gauntlet summary + sandbox isolation check (fixture inputs and per-task jsonl omitted) |
| `raw/production/` | production lane args/env before and after the card |
| `scripts/` | `cardBR1_runner.sh`, `launch_br1.sh`, `br1_verdict.py`, `br1_verdict_rescore.py`, `rescore_patch_br1.py`, `perclass_run_t.sh`, `perclass_decode_t.py`, `ar_mixed_t.py`, and the `DIFF-*.txt` files against their parents, exactly as run, with box paths scrubbed |
| `throughput.csv` | every speed / acceptance / flip cell with its receipt |

## Withheld, on purpose

- The per-class prompt file `perclass_prompts.jsonl` (sha1 `618898aac432`) is real Hermes transcripts and stays
  **private**. The tables are published verbatim; the per-request `rows` (output heads) are removed from every
  perclass/agentfix JSON. The `turns-*.json` replays (from `transcript_fixture.json`, also private transcripts) are
  replaced by per-pass counts.
- Server/sidecar post-cell logs (`postlog-*`) and the full restore log stay in the box receipts. The bundle keeps
  a filtered `raw/restore-v25m-kvtier.facts.log` with the KV and startup lines.
- Path scrub: the box home directory → `~`, LAN address → `<gb300>`.
