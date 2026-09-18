# v18-cgsizes — token-correct CUDA graph capture sizes vs v17 (2026-09-18)

Worker: grok-4.6 · xai-oauth. Axis: `--cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128` (token counts) on the v17 many-seat profile (seqs 24, lpt 6144, ksched `[[1,4,5],[5,24,1]]`). Hook/rowmap/off60/util 0.97/batched 8192/image unchanged.

This run corrects v17b's mistaken list `1 2 4 8 12 16 24` (written as sequence counts). With DSpark k=1 at ≥5 seqs a step is 2×seqs tokens, so that list left >12 seqs uncaptured.

## Box / boot

- Pre 12:44:07 CDT: only `dsv41-vllm-v17-seqs24-BOUND-REF` on :30006. GLM stopped-and-kept. :30003 down. No iwbench. GPU1 245538/256703 MiB. MemAvailable 131386368 kB.
- Stop-and-keep v17 12:44:13. Drop caches (MemAvailable 488685376 kB). Launch `dsv41-vllm-v18-cgsizes-EXP` 12:44:16 CDT.
- Inspect Args: `--max-num-seqs 24`, `--cudagraph-capture-sizes` exactly those 15 values, lpt 6144, ksched `[[1,4,5],[5,24,1]]`.
- Hook: 40 layers `hot_w1=(295,…)` `cold_w1=(89,…)`; `HBM_expert=206.61GiB` `pinned_expert=62.33GiB`.
- Graphs: 36 s / **1.30 GiB**, then 14 s / **1.10 GiB** (v17 default graphs 2.33 + 1.78 GiB).
- KV **2,502,950 tok** (2.39× at 1M). Bar ≥2.4M — pass. No KV gate.
- Autotune **MISS** hash `62426808ae2b0ddbdfbf2278702ac4e367078b9884817c169ddee40495c4e6a6` (dir 103a). Tune 17:51:30–18:07:18 UTC (~16 min; 189 new, 0 previous). Capture sizes *did* change the kernel-shape hash (brief expected a hit; miss is not a failure). Bind 13:07:48 CDT (~23.5 min wall).
- Serving log: 0 matches `ValueError:|Traceback|ERROR|RuntimeError:`.

## Idle cold_prefill_probe (v18, SIZES 8k/32k/128k, N=2)

| target | prompt_tok | ttft_s | mean tok/s |
|---|---|---|---|
| 8000 | 6524 | 0.45 / 0.38 | 15704 |
| 32000 | 25932 | 1.20 / 1.23 | 21325 |
| 128000 | 103606 | 4.58 / 4.58 | 22657 |

v17 idle: 12.1k / 17.5k / 23.3k tok/s, 4.46 s at 103.6k. Idle 128k **−2.6%** vs 23263 (bar ≥ −5%).

## Same-window: v18A → v17ctl → v18B → v15ctl3

Instrument: `$BOX_HOME/iwyzer/iwbench.py` `CHARS_PER_TOKEN=3.894` `BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none THINKING=0`. Sequence: mix 8/120 (discard), mix 16/420, mix 24/420, coldload 5, tools 16, `knee.sh` ×2. Tags **v18A / v17ctl / v18B / v15ctl3**. Controls = stop-and-keep + `docker start` named REF (hash-hit, ~5–6 min).

| metric | v18A | v18B | v18 mean | v17ctl | bar | result |
|---|---|---|---|---|---|---|
| KV tokens | 2.503M | 2.503M | 2.503M | 1.786M | ≥2.4M | PASS |
| graphs GiB | 1.30+1.10 | — | 1.30+1.10 | 2.33+1.78 | report | — |
| C24 warm agent TTFT p50 s | 0.545 | 0.561 | **0.553** | 0.775 | ≤5 | PASS |
| C16 warm agent TTFT p95 s | 1.820 | 1.897 | **1.859** | 1.898 | ≤ v17+0.5 (2.398) | PASS |
| C16 chat TTFT p95 s | 14.766 | 11.272 | **13.019** | 20.054 | ≤ v17×1.1 (22.059) | PASS |
| C16 prose decode p10 tok/s | 15.6 | 15.9 | **15.75** | 15.3 | ≥ −5% | PASS (+2.94%) |
| Cold 120k /16 p95 s | 9.158 | 9.155 | **9.157** | 8.867 | ≤ v17×1.1 (9.754) | PASS |
| Idle 128k tok/s | 22657 | — | 22657 | 23263 | ≥ −5% | PASS (−2.6%) |
| knee C1 agg | 176.3/177.1 | 117.4/98.6 | **142.34** | 164.2/164.8 | ≥ −1.5% | FAIL (−13.48%) |
| knee C8 agg | 680.9/681.8 | 587.0/585.3 | **633.77** | 651.2/653.8 | ≥ −1.5% | FAIL (−2.87%) |
| knee C16 agg | 998.2/994.5 | 732.7/746.8 | **868.05** | 952.0/950.8 | ≥ −1.5% | FAIL (−8.76%) |
| tools | 64/64 | 64/64 | — | 64/64 | ≥ v17 count | PASS |
| host Δ | MemAvailable after v18 bind 129878336 kB vs v17-serving 131386368 kB (−1.44 GiB avail) | | | | ≤2 GiB extra | PASS |
| serving-log ERROR/Traceback | none | none | | none | none | PASS |

v18A knees alone would have been C1 +7.4% / C8 +4.4% / C16 +4.7% vs v17ctl. v18B mix (C16 prose p10 15.9, C24 warm p50 0.561, C16 warm p95 1.897) did **not** collapse; only `knee.sh` did (C1 117/99, C12 442/463). Two-window mean is the bar; it fails. Do not average away the second window.

## v15ctl3 (report only — settle v17 vs v15)

v15ctl3 C16 prose p10 **16.5**; knee C8 **696.8/699.3 mean 698.0**.

- v17ctl C16 prose p10 15.3 vs v15ctl3 16.5 = **−7.3%**. `results/v17/` vs v15ctl2 was −11.85% (15.25 vs 17.3). Direction holds; magnitude smaller on this window.
- v17ctl knee C8 652.5 vs v15ctl3 698.0 = **−6.5%**. `results/v17/` was −7.06% vs v15ctl2. The C8 loss vs v15 is real across two control windows.

v15ctl3 C24 warm p50 27.344 s (slot queue at max-num-seqs=16); tools 64/64.

## Leave-up

Not all bars (knee C1/C8/C16). Stop-and-keep v18 renamed `dsv41-vllm-v18-cgsizes-EXP-KNEE-FAIL`. `docker start` v17 `-BOUND-REF`. Never rm. GLM / :30003 untouched.

KV recovery worked (1.786M → 2.503M) and capture-size decode cost did **not** show in mix/idle/tools. The two-window knee mean failed because v18B `knee.sh` collapsed; that is the recorded reason, not a mix-path regression.

## Files

`iw-v18A.jsonl`, `iw-v17ctl.jsonl`, `iw-v18B.jsonl`, `iw-v15ctl3.jsonl`, `run-*.log`, `knee-*.json`, `cold_prefill_probe.out`, `boot-log-excerpt.txt`, `bind-status.txt`, `stop-drop-launch.log`.
