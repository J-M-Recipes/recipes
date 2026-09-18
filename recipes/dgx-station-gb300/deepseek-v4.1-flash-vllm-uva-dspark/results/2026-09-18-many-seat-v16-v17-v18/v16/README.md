# v16-sched — scheduler-only candidate vs v15 (2026-09-18)

Worker: grok-4.6 · xai-oauth. Axis: `--max-num-seqs 32`, `--long-prefill-token-threshold 2048`, k-schedule `[[1,4,5],[5,32,1]]`. Hook/rowmap/off60/util 0.97/batched 8192/image unchanged.

## Box / boot

- Pre: only `dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` on :30006. GLM stopped-and-kept. :30003 down. GPU1 253202/256703 MiB. MemAvailable 126469056 kB.
- Stop-and-keep v15. Drop caches (MemAvailable 486310976 kB). Launch `dsv41-vllm-v16-sched-EXP` 05:53:15 CDT.
- Hook: 40 layers `hot_w1=(295,…)` `cold_w1=(89,…)`; `HBM_expert=206.61GiB` `pinned_expert=62.33GiB`.
- Graphs: 50 s / 2.85 GiB, then 32 s / 2.44 GiB. KV **1,387,235 tok** (2.91 GiB, 1.32× at 1M). Bar was ≥2.0M — fail. No OOM → no SEQS=24 fallback.
- Autotune **MISS** hash `b8c6fdcc63725861ce9b8da8148365388cbf8ad686ea96ee3ba0cea9fa8dfc51` (dir 103a). Tune 11:00:46–11:16:55 UTC (~16 min). Bind 06:17:44 CDT (~24 min wall).
- Restart binds hash-hit (~6 min; autotune 6 s). No `ValueError:|Traceback|ERROR|RuntimeError:` in v16 serving log.

## Idle cold_prefill_probe (v16, SIZES 8k/32k/128k, N=2)

| target | prompt_tok | ttft_s | mean tok/s |
|---|---|---|---|
| 8000 | 6538 | 1.33 / 0.59 | 7987 |
| 32000 | 25851 | 2.42 / 2.28 | 11026 |
| 128000 | 103602 | 8.97 / 8.95 | 11580 |

v15 idle ref (Milo): 16.8k / 22.3k / 22.0k tok/s, 4.72 s at 103.6k. Idle 128k −47% vs that ref (>25%). LPT=4096 follow-up **not run**: warm bar already passed, but KV 1.39M, C16 prose p10 −29%, and chat p95 mean 10.28 s cannot be rescued by LPT-only; ALL-bars leave-up was already impossible.

## Same-window: v16a → v15ctl → v16b

Instrument: `$BOX_HOME/iwyzer/iwbench.py` `CHARS_PER_TOKEN=3.894` `BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva`. Sequence: mix 8/120 (discard), mix 16/420, mix 24/420, coldload 5, tools 16, `knee.sh` ×2.

| metric | v16a | v16b | v16 mean | v15ctl | bar | result |
|---|---|---|---|---|---|---|
| C16 warm agent TTFT p95 s | 1.313 | 1.363 | **1.338** | 1.916 | ≤2.0 (hard ≤4.0) | PASS |
| C16 chat TTFT p95 s | 9.009 | 11.552 | **10.281** | 18.791 | ≤10 | FAIL |
| Cold 120k /16 p95 s | 15.041 | 15.811 | **15.426** | 20.581 | ≤15 | FAIL |
| C24 warm agent p50 s | 1.723 | 5.510 | **3.616** | 27.093 | single digits | PASS |
| C16 prose decode p10 tok/s | 11.7 | 12.3 | **12.0** | 16.8 | ≥ −10% | FAIL (−28.6%) |
| knee C1 agg | 174.0/174.4 | 153.6/154.1 | **164.0** | 153.2/153.4 | ≥ −1.5% | PASS (+7.0%) |
| knee C8 agg | 733.0/733.9 | 704.4/705.1 | **719.1** | 699.6/704.3 | ≥ −5% | PASS (+2.4%) |
| knee C16 agg | 1051.9/1048.6 | 1002.2/1001.6 | **1026.1** | 952.2/950.9 | ≥ −5% | PASS (+7.8%) |
| tools | 63/64 | 64/64 | — | 61/64 | 64/64 | FAIL (v16a) |
| KV tokens | 1.387M | 1.387M | 1.387M | 2.262M | ≥2.0M | FAIL |
| host Δ | MemAvailable after v16 bind 128048768 kB vs v15-serving 126477632 kB | | | | ≤2 GiB extra | PASS |
| serving-log ERROR/Traceback | none | none | | none | none | PASS |

C24 is the intended win: v15ctl warm p50 27.1 s vs v16 mean 3.6 s (slot queue at max-num-seqs=16). C16 warm was already ~1.9 s on this quieter same-window v15ctl (fund-post 8.54 s was a different mix/load). Decode cost is real: prose p10 −29%.

## Leave-up

Not all bars. Stop-and-keep v16 renamed `dsv41-vllm-v16-sched-EXP-BARS-FAIL`. `docker start` v15 BOUND-REF. Never rm. GLM / :30003 untouched.

## Files

`iw-*.jsonl`, `run-*.log`, `knee-*.json`, `cold_prefill_probe.out`, `boot-log-excerpt.txt`, `boot-stamp.txt`, `bind-status.txt`. Launcher: `scripts/launch-v16-sched.sh` (SEQS=32 LPT=2048).
