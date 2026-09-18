# v17-seqs24 — max-num-seqs 24 vs v15 (2026-09-18 resume)

Worker: grok-4.6 · xai-oauth. Axis: `--max-num-seqs 24`, `--long-prefill-token-threshold 6144`, k-schedule `[[1,4,5],[5,24,1]]`, default CUDA graphs. Hook/rowmap/off60/util 0.97/batched 8192/image unchanged.

This run is the **resume** after Milo stopped the first v17 worker. Two errors were **in Milo's brief**, not the worker:

1. KV gate (≥ 1.8M) was arbitrary. v17 booted at **1,786,288 tokens** (1.70× at 1M), 14K under it. Gate withdrawn. This container is the one measured.
2. Fallback `--cudagraph-capture-sizes 1 2 4 8 12 16 24` was written as sequence counts; the flag takes **token** counts. With DSpark k=1 at ≥ 5 seqs a step is 2×seqs tokens, so everything above 12 seqs ran uncaptured. That container is stopped-and-kept as `dsv41-vllm-v17b-seqs24-cg-EXP-TOKENSIZES-UNCAPTURED-FAIL`. Its partial `iw-v17a.jsonl` / `run-v17a.log` are **v17b-uncaptured data** — see `v17b-uncaptured/`, do not mix into the v17 table.

**Real finding from v17b, kept:** trimming capture sizes cut graph memory 2.33+1.78 GiB → 0.61+0.46 GiB and KV 1.79M → **2.96M** tokens (more than v15's 2.26M). A *correct* token list is a future lever; not this run.

## Box / bind (resume)

- Pre: only `dsv41-vllm-v17-seqs24-EXP` on :30006 (Milo `docker start` ~09:39 CDT). GLM stopped-and-kept. :30003 down. No iwbench. Do not launch anything new.
- Inspect: `--max-num-seqs 24`, `--long-prefill-token-threshold 6144`, ksched `[[1,4,5],[5,24,1]]`.
- Hook: 40 layers `hot_w1=(295,…)` `cold_w1=(89,…)`; `HBM_expert=206.61GiB` `pinned_expert=62.33GiB`.
- KV **1,786,288 tok** (1.70× at 1M). Graphs (original boot) 2.33 then 1.78 GiB; resume restart 2.33 then 2.08 GiB.
- Resume restart hash-hit: Loaded 189 / 0 new, autotune ~6 s, `/v1/models` 09:44 CDT (~5 min). No `ValueError:|Traceback|ERROR|RuntimeError:` in serving log.

## Idle cold_prefill_probe (v17, SIZES 8k/32k/128k, N=2)

| target | prompt_tok | ttft_s | mean tok/s |
|---|---|---|---|
| 8000 | 6519 | 1.01 / 0.37 | 12054 |
| 32000 | 25984 | 2.01 / 1.18 | 17486 |
| 128000 | 103614 | 4.46 / 4.46 | 23263 |

v15 idle ref (Milo): 16.8k / 22.3k / 22.0k tok/s, 4.72 s at 103.6k. Idle 128k **+5.7%** vs 22.0k (bar ≥ −5%).

## Same-window: v17A → v15ctl2 → v17B

Instrument: `$BOX_HOME/iwyzer/iwbench.py` `CHARS_PER_TOKEN=3.894` `BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none THINKING=0`. Sequence: mix 8/120 (discard), mix 16/420, mix 24/420, coldload 5, tools 16, `knee.sh` ×2. Tags capital **v17A / v15ctl2 / v17B**.

| metric | v17A | v17B | v17 mean | v15ctl2 | bar | result |
|---|---|---|---|---|---|---|
| C24 warm agent TTFT p50 s | 1.870 | 0.785 | **1.328** | 27.410 | ≤ 5 | PASS |
| C24 prose decode p10 tok/s | 8.9 | 8.8 | **8.85** | 12.5 | report only | — |
| C16 warm agent TTFT p95 s | 1.885 | 1.922 | **1.903** | 1.760 | ≤ v15+0.5 (2.260) | PASS |
| C16 chat TTFT p95 s | 14.354 | 8.614 | **11.484** | 14.958 | ≤ v15×1.1 (16.454) | PASS |
| C16 prose decode p10 tok/s | 15.6 | 14.9 | **15.25** | 17.3 | ≥ −5% | FAIL (−11.85%) |
| Cold 120k /16 p95 s | 8.778 | 8.977 | **8.878** | 15.773 | ≤ v15×1.1 (17.350) | PASS |
| Idle 128k tok/s | 23263 | — | 23263 | 22000 ref | ≥ −5% | PASS (+5.7%) |
| knee C1 agg | 164.5/165.0 | 164.2/164.6 | **164.57** | 153.4/153.4 | ≥ −1.5% | PASS (+7.28%) |
| knee C8 agg | 658.3/658.4 | 655.2/641.7 | **653.4** | 703.1/703.0 | ≥ −5% | FAIL (−7.06%) |
| knee C16 agg | 954.7/956.1 | 950.8/951.0 | **953.15** | 957.1/953.3 | ≥ −5% | PASS (−0.21%) |
| tools | 64/64 | 64/64 | — | 63/64 | ≥ v15 count | PASS |
| KV tokens | 1.786M | 1.786M | 1.786M | 2.262M | gate withdrawn | — |
| host Δ | MemAvailable after v17 resume bind 131072768 kB vs v16-era v15-serving 126477632 kB (v17 more free, not extra) | | | | ≤2 GiB extra | PASS |
| serving-log ERROR/Traceback | none | none | | none | none | PASS |

C24 is the intended win: v15ctl2 warm p50 27.4 s vs v17 mean 1.328 s (slot queue at max-num-seqs=16). C16 prose p10 −12% with lpt back at 6144 is a **seqs finding** (v16 was −29% with lpt 2048). Knee C8 −7% also fails the −5% bar.

## Leave-up

Not all bars. Stop-and-keep v17 renamed `dsv41-vllm-v17-seqs24-EXP-BARS-FAIL`. `docker start` v15 BOUND-REF. Never rm. GLM / :30003 untouched.

## Files

`iw-v17A.jsonl`, `iw-v15ctl2.jsonl`, `iw-v17B.jsonl`, `run-*.log`, `knee-*.json`, `cold_prefill_probe.out`, `boot-log-excerpt.txt`, `v17b-uncaptured/`.
