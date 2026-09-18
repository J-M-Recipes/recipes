# Many Seat — v16 / v17 / v18 (2026-09-18)

Scheduler-only campaign on top of v15 (pin-hot-experts hook, off60, util 0.97, 1M ctx). Trigger: the [35-seat fund workload post](https://al-engr.com/gb300-35-seat-fund-workload.html) measured this lane at **warm agent-turn p95 8.54 s at 16 streams and p50 27 s at 24**. Every boot changed one axis and was measured same-window against the incumbent on the fund harness (`iwbench.py`: mix 16/420 s, mix 24/420 s, coldload 5, tools 16×4) plus `knee.sh` ×2. Operator runs by grok-4.6 (xai-oauth) from Milo's briefs in [`briefs/`](briefs/); Milo re-checked the v18 knee by hand on an idle box.

## Diagnosis first

Raw prefill was **not** the problem. Idle cold prefill on v15 (`cold_prefill_probe.py`, nonce-first, `max_tokens 1`): 6.5K 16.8K tok/s · 26K 22.3K · 104K 22.0K (4.72 s) · 207K 20.5K · 405K 18.5K; 52K with eight decoders live 21K. The serving log during the fund run showed `Running: 16` pinned at `--max-num-seqs 16`, `Waiting: 10–12` at C24, **KV usage peaking at 16 %**. The V1 scheduler in this build (`v1/core/sched/scheduler.py:598`) serves `running` FCFS with `--long-prefill-token-threshold` as a per-request per-step chunk cap; `max_num_partial_prefills` does not exist in it.

## Ladder

| boot | change vs v15 | KV tokens | graphs GiB | verdict |
|---|---|--:|--:|---|
| v16 | seqs 32 + lpt 2048 + ksched `[[1,4,5],[5,32,1]]` | 1,387,235 | 2.85 + 2.44 | FAIL — C24 queue fixed (warm p50 27.1 → 3.6 s) but idle prefill −47 %, prose p10 −29 %, KV too low. Two knobs, opposite directions. |
| v17 | seqs 24 + ksched `[[1,4,5],[5,24,1]]` (lpt 6144) | 1,786,288 | 2.33 + 1.78 | promoted midday: C24 warm p50 27.4 → 1.33 s, cold 120K p95 15.8 → 8.9 s, knee C1 +7 %; KV cost from default 24-seq graphs |
| v17b | v17 + `--cudagraph-capture-sizes 1 2 4 8 12 16 24` | 2,955,040 | 0.61 + 0.46 | FAIL — **sizes are tokens**; above 12 seqs decode ran uncaptured (C16 prose 9.6 tok/s/stream). Kept as the KV-lever proof. |
| **v18** | v17 + `--cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128` | **2,502,950** | 1.30 + 1.10 | **reference** — knee within noise of v17, KV +40 % |

Two errors in Milo's own briefs are on the record in [`v17/README.md`](v17/README.md): a KV gate 14K tokens too tight, and the v17b list written in sequences.

## v18 vs controls

Knee (`knee.sh`, prose, T=0, 192 tok, agg tok/s; two invocations per window):

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| v18 (idle re-check, Milo) | 171.8 / 172.1 | 653.8 / 656.9 | 747.5* / 944.6 |
| v17 (idle, same window) | 164.9 / 165.1 | 640.6 / 658.7 | 878.1 / 949.7 |
| v15ctl3 (same afternoon, Grok) | 152.7 / 152.9 | 696.8 / 699.3 | 942.9 / 943.7 |
| v18A (Grok window) | 176.3 / 177.1 | 680.9 / 681.8 | 998.2 / 994.5 |
| v18B (Grok window) | 117.4 / 98.6 | 587.0 / 585.3 | 732.7 / 746.8 |

\* one internal rep at 552 (v17 has an 878 the same way). **v18B is excluded**: it ran across a worker tool-cap resume with `iwbench` still live; the idle re-check does not reproduce it (C1 172 both passes). Net vs v15: **C1 +12.6 %, C8 −6.2 %, C16 flat.** The C8 loss is consistent across two v15 controls and unexplained (new autotune hash suspected).

Fund harness (means of v18A/v18B):

| metric | v18 | v17ctl | v15ctl3 |
|---|--:|--:|--:|
| C16 warm agent-turn TTFT p95 | **1.86 s** | 1.90 | 1.88 (morning run: 8.54) |
| C16 chat (10.7K prompt) TTFT p95 | 13.0 s | 20.1 | 17.2 (morning: 28.2) |
| C24 warm agent-turn p50 | **0.55 s** | 0.78 | **27.3** |
| Cold 120K under 16 streams p95 | **9.2 s** | 8.9 | 19.1 |
| Interference: 16 streams outside → during | 32 → 5 (−85 %) | 31 → 5 | 30 → 17 (−44 %, queue-bound) |
| C16 prose decode p10 / mean | 15.75 / 21.7 | 15.3 / 19.8 | 16.5 / 21.2 |
| C8 prose p10 / mean (420 s window on the promoted lane) | 28.0 / 40.1 | — | (v15 morning 20 / 27) |
| Tools 16 × 4 × 12 | 64/64 · 64/64 | 64/64 | 64/64 |
| Idle cold prefill 104K | 22.7K tok/s | 23.3K | 22.0K |

The harness swings ±20 % on p95 run to run (v15 C16 warm p95 read 8.54 s in the morning and 1.88 s in the afternoon); the queue-vs-no-queue gap at C24 is 50×.

## Files

`v16/`, `v17/`, `v18/`: per-window `iw-*.jsonl` receipts, `run-*.log` harness summaries, `knee-*.json`, boot-log excerpts, worker READMEs. `v17/v17b-uncaptured/` holds the wrong-list receipts, labelled. `briefs/`: the four operator briefs as issued (including the errors). `launch-many-seat.sh`: the launcher (env `SEQS LPT KSCHED CGSIZES NAME`; defaults = v18). `throughput.csv`: every knee row plus headline harness rows. Box paths and LAN addresses scrubbed (`$BOX_HOME`, `<station>`).
