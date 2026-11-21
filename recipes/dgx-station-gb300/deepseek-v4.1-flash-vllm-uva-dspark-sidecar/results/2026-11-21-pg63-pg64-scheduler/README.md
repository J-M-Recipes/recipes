# Pg63 + Pg64 (2026-11-21): prefill cadence and prefill yield on the GB300 sidecar lane — PROMOTED, experimental, lane stopped-and-kept

**Verdict.** Two scheduler-only changes were promoted on this lane on 2026-11-21, each after pre-registered bars passed on
two separate boots against same-window controls:

| config | = | promoted (CST) | what it buys | what it costs |
|---|---|---|---|---|
| **Pg63** | Pg62 + `patches/cadence3.diff` (env-gated `vllm/v1/core/sched/scheduler.py` overlay) + `VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192` | 09:31:08 ([log](promote/promote-pg63-20261121-0923.log)) | a reply streaming beside a 413K-token prompt decodes at **33 tok/s instead of 12–13** (2.7×); beside a 1M prompt **29–30 instead of 10–11**; a short chat behind a 413K prompt gets its first token in **0.33–0.36 s instead of 0.49 s** | the long prompt itself finishes **+3.0 … +7.9 % later while a reply streams beside it** (trace-d by arrival time); ITL of that reply becomes bimodal |
| **Pg64** | Pg63 + `patches/yieldcad.diff` (= cadence3 + Christopher Owen's prefill-yield mechanism, spark-ds41f patch 0545, ported to stock v0.31.0) + `VLLM_PREFILL_YIELD=short VLLM_PREFILL_YIELD_LIMIT_S=2` | 12:08:42 ([log](promote/promote-pg64-20261121-1201.log)) | a short chat arriving while a 413K prompt is being read gets its first token in **0.16–0.22 s instead of 0.27–0.58 s** (phase-locked probe P1, three boots: 0.68× / 0.49× / 0.55× the pooled control median) | long prompt **+0.2 … +0.8 %** on P1; an **8,192-token prompt is not helped** (P2 1.10–1.12 s vs 1.07 s); Y2's C1 read −2.5 % against a −2 % bar (Y1 −0.5 %, Y3 −0.7 %) |

Everything else is Pg62: same image (`vllm/vllm-openai:v0.31.0`), same Garner engine tree, same args (`--long-prefill-token-threshold 6144`,
`--max-num-batched-tokens 8192`), same KV pool (11,696,582 tokens, every boot log), same speculation. Each promotion verified the overlay
sha256 inside `ds41` against the overlay source and the env pairs in `docker inspect`.

**Lane state when this bundle was written (2026-11-21 ~15:00 CST):** Pg64 is the current config, but the lane is **stopped-and-kept**
since **2026-11-21 14:49:50 CST** at James's request (GPU freed; `garner/pg64-work` intact, [`promote/lane-owner-20261121.txt`](promote/lane-owner-20261121.txt)
line 53). Restore = `WORK=~/dsv41/garner/pg64-work MODEL_DIR=… DS41_HOME=~/dsv41/garner/home bash launch-pg61.sh up`. A measurement card
(KL1) took the dark box at 14:57. Status stays `experimental`.

**What this round did not do:** no 48 h soak on Pg63 or Pg64 (Pg64's soak clock started 12:08:21 and stopped at 14:49:50 — 2 h 41 min); **no T2/T3
quality suite** (needle-checksum 500, GSM8K-1319, BFCL held-out, MMLU-Pro) on Pg63 or Pg64 — the fidelity evidence here is the card battery's
qeval (75 auto-scored tasks), one 3-needle probe at 1M, greedy same-boot 40/40, a 64-request structured-JSON burst and the DSML live probe, on
every arm; no teacher-forced bracket; no temperature > 0 instrument; neither rollback script (`rollback-pg64-to-pg63.sh`, `rollback-pg63-to-pg62.sh`)
has been run; no caller-probe table on Pg63/Pg64.

## What a scheduler change does and does not touch

Both overlays change **which requests get which engine step**, nothing else. Chunk size (6,144), chunk boundaries, kernels, weights, KV,
sampling and the sidecar are untouched. With the env vars unset every added code path is skipped and `schedule()` is the stock v0.31.0 code
([`patches/cadence3.diff`](patches/cadence3.diff) 92 lines, [`patches/yieldcad.diff`](patches/yieldcad.diff) 370 lines, both against the stock
file sha256 `f820bee4…`, [`patches/SHA256SUMS`](patches/SHA256SUMS)). Batch composition per step does change, and this lane is already not
bit-identical across batch compositions (det_check below: 0 of 32 concurrent greedy requests reproduce their logprobs on the Pg63 controls too).

- **Prefill cadence (Pg63).** The synchronous scheduler gives a running reply one step per 6,144-token prompt chunk (~167 ms at 413K, ~219 ms at
  1M), so it decodes at ~13 tok/s beside a long prompt. The overlay reuses stock v0.31.0's data-parallel prefill-throttle path
  (`throttle_prefills` → `defer_prefills`): while a running request still has more than `MIN_REMAINING` (8,192) uncomputed prompt tokens, two
  of every three `schedule()` calls go decode-only. Short prompts and cached-prefix suffixes are never deferred, and with no long prompt in
  flight the tick resets and the stock path runs. Engine log: `xfer prefill cadence enabled: N=3 min_remaining=8192` at boot, then
  `deferred N of M steps` counters ([`raw/logs/driver.log`](raw/logs/driver.log)).
  **Upstream status:** vLLM applies `--prefill-schedule-interval N` to a single engine from v0.32.1 on
  ([#54627](https://github.com/vllm-project/vllm/pull/54627), merged 2026-10-15; reworked on main by
  [#66224](https://github.com/vllm-project/vllm/pull/66224)); on v0.31.0 and v0.32.0 it only takes effect with data parallelism. So the
  cadence half of this overlay is a backport. What upstream does not have is the long-only gate (`MIN_REMAINING` plus ticking only while a
  long prefill is in flight): upstream holds back every prefill, which is the X1 arm that failed the 32-user bar. On a rebase to ≥ v0.32.1, use
  the flag and carry only the gate.
- **Prefill yield (Pg64).** Owen's `--prefill-yield short`: a long prompt's next chunk gives its step to waiting prompts whose prefill finishes
  within one chunk; a long prefill that has yielded for `LIMIT_S` (2 s) takes its chunk regardless. Owen's sink-rejection half of 0545 is not
  ported. The port is the 2026-11-16 desk port, merged onto cadence3 with `patch -p1` (no rejects).

## Timeline (from the box's LANE-OWNER ledger and the promotion logs)

| CST | event | receipt |
|---|---|---|
| 04:51–04:59 | A1: live Pg62 battery + post suite (control before) | `raw/battery/gb300-pg62-A1/`, `raw/post/gb300-pg62-A1/` |
| 04:59–05:52 | Card X1: B (N=2) → C (N=3) → GR (#59131+#63540) → restore Pg62 → A2 | `cards/CARD-X1-SPEC.md`, `cards/X1-table.md` |
| 06:03–06:48 | Card X2: D1, D2 (cadence2, long-only) → restore → A3 | `cards/CARD-X2-SPEC.md` |
| 06:48–07:33 | Card X3: E1, E2 (cadence3, gated) → restore → A4 | `cards/CARD-X3-SPEC.md` |
| 07:37–08:04 | X3b diagnostic: fresh boots E3 (cadence on), F (overlay, env off), A5 (Pg62), each C32 ×3 cold → 1M trace → C32 ×2 | `raw/logs/driver.log` lines 36–60, `raw/c32x5/bench-diag-*`, `raw/logs/engine-diag-*.log.gz` |
| 08:06–09:03 | Card X4: A6 → G1 → G2 → restore → A7, each + C32×5 | `cards/CARD-X4-SPEC.md`, `cards/X4-table.md`, `raw/x4-c32.json` |
| 09:08–09:22 | **Pg63 promotion, attempt 1: false FAIL** — the script's `docker logs ds41 \| grep -q` under `pipefail` SIGPIPEd and reported "cadence not enabled"; the script's failure path ran `xfer_arm.sh restore` and **Pg62 was back serving at 09:22:24** | [`promote-pg63-20261121-0908.log`](promote/promote-pg63-20261121-0908.log) |
| 09:23–09:31 | **Pg63 promoted** (check rewritten as `grep -ac`; overlay sha256 in-container equal; env present; smoke OK; watchdog `trip: null`) | [`promote-pg63-20261121-0923.log`](promote/promote-pg63-20261121-0923.log), [`promote-pg62-to-pg63.sh`](promote/promote-pg62-to-pg63.sh) |
| 10:01–11:16 | Card X5 on Pg63: K1 → D1 ✗ → Y1 → D2 ✗ → Y2 → restore Pg63 → K2 | `cards/CARD-X5-SPEC.md`, `cards/X5-table.md`, `raw/x5.json` |
| 11:33–12:00 | Card X6: Y3 (third yield boot, James's call) → restore Pg63 | `cards/CARD-X6-SPEC.md`, `raw/x6.json` |
| 12:01–12:08 | **Pg64 promoted** | [`promote-pg64-20261121-1201.log`](promote/promote-pg64-20261121-1201.log), [`promote-pg63-to-pg64.sh`](promote/promote-pg63-to-pg64.sh) |
| 14:49:50 | **Pg64 stopped-and-kept** at James's request; GPU freed | `promote/lane-owner-20261121.txt` line 53 |

Every arm boot is a copy of the production tree plus the overlay file(s) and env, built by [`scripts/xfer_arm.sh`](scripts/xfer_arm.sh)
(Pg62 base) / [`scripts/xfer_arm2.sh`](scripts/xfer_arm2.sh) (Pg63 base); the arm script refuses unless `diff -rq` against production shows
exactly the expected files, and after `up` it checks `docker inspect` args/env and the overlay sha256 inside the container. The arm `serve.sh`
differs from production in two no-op-unless-set hooks: `DS41_ARM_ENV` (extra server env) and `DS41_LPT` (threshold, held at 6144); `serve.sh.arm2`
adds `DS41_SIDECAR_ENV`. Those are the only `tree/` diffs (`tree diffs=1` in every op log).

## Instruments

- **Battery** (`battery.py` from `~/hermes/work/dsv41-5way-2026-10-26/`, same harness as the 4× Spark r7 round): `c1` / `c16` = `v41bench_tf.py`
  prompt set v1, T=0, thinking off, median of 3 reps; `prefill` = `prefill_reps.py`, unique nonce, `max_tokens 1`, median of 3 (1 at 1M);
  `needle` = 3 passphrases at 10/50/90 % depth in 1M; `mixed_short` = ~18-token chat TTFT **to the first answer token with thinking on**
  (diagnostic — it includes thinking decoded at the shared rate); `qeval` = 75 auto-scored tasks.
- **Post suite** ([`scripts/tools/post_arm_gb300.sh`](scripts/tools/post_arm_gb300.sh)): `mixed_trace.py` traces — `trace-s-*` (32-token probe,
  thinking on, arriving 6.6/7.5/8.2/9.0 s into a cold 413K prefill; "true TTFT" = first streamed delta), `trace-d-*` (400-token reply, thinking
  off, arriving 2/4/6 s in; `decode_during_prefill_tok_s` = reply tokens between its first token and the long prompt's completion),
  `trace-m1m-4.0` (same beside a 1M prompt); one single-rep C32 (`v41bench_tf.py --levels 32`); gates; greedy same-boot ×2; a 64-request JSON burst;
  the DSML live probe. The `413K time while shared` row in `cards/*-table.md` mixes the short-probe and reply traces; read `trace-d` `long_done`
  by arrival for the cadence cost (the table below splits them).
- **C32×5** (X4 onward): five back-to-back C32 reps after the post suite, medians compared ([`scripts/driver_x4.py`](scripts/driver_x4.py)).
  Why: the X3b diagnostic showed the **first C32 after any fresh boot reads ~2,700 tok/s against ~3,950 warm, on Pg62 itself** (`diag-A5-cold-r1`
  2,695.7; `diag-E3-cold-r1` 2,670.5; `diag-F-cold-r1` 2,859.3), and 0.3–0.74 s single-category stalls recur at random on all three arms
  ([`raw/logs/driver.log`](raw/logs/driver.log) lines 39–60). The single-rep C32 inside the post suite is therefore a noisy instrument; the
  X1–X3 C32 verdicts below stand as recorded under it.
- **Phase-locked probe** (X5/X6, [`scripts/driver_x5.py`](scripts/driver_x5.py)): P1 = a short chat (thinking off) sent at 0.8 × that rep's own
  solo cold-413K time while a cold 413K prompt is read, ×5; P2 = the same with an 8,192-token prompt (not single-chunk under LPT 6144, so yield
  must not help it), ×2. `det_check.py` = 32 concurrent greedy requests with logprobs, run twice.
- The post suite's `gates.py` smoke check fails on every GB300 arm including controls (a Spark-shaped check); it is not a gate on this lane.

## Verdict tables

Control columns are plain; arm columns are bold with the ratio or % against the control mean for that card. C1/C16 are the battery medians of 3;
cold prefill is the median of 3 reps (1 at 1M). Bars are the ones pre-registered in `cards/CARD-X*-SPEC.md` before each card's first boot.

### Card X1 — window 1: cadence N=2 (B), N=3 (C), upstream #59131+#63540 (GR) on Pg62 (controls A1 before, A2 after)

| cell | gb300-pg62-A1 | gb300-pg62-A2 | **gb300-B** | **gb300-C** | **gb300-GR** |
|---|--:|--:|--:|--:|--:|
| decode during 413K prefill, tok/s (trace-d median of 3) | 12.6 | 12.8 | **23.2 (1.83×)** | **34.6 (2.72×)** | **12.8 (1.01×)** |
| decode during 1M prefill, tok/s (1 trace) | 10.2 | 11.1 | **14.4 (1.35×)** | **30.2 (2.84×)** | **10.0 (0.94×)** |
| 413K prompt time while a 400-token reply streams, s (arrivals 2 / 4 / 6 s) | 11.39 / 11.15 / 11.00 | 11.21 / 11.15 / 11.01 | **11.82 / 11.51 / 11.23** | **12.17 / 11.87 / 11.40** | **11.21 / 11.09 / 10.90** |
| 413K prompt time while a 32-token probe runs, s (median of 4) | 10.85 | 10.91 | **10.93 (+0.5 %)** | **10.90 (+0.2 %)** | **10.79 (-0.8 %)** |
| 1M prompt time while shared, s | 37.54 | 37.43 | **38.58 (+2.9 %)** | **37.37 (-0.3 %)** | **36.88 (-1.6 %)** |
| true first token, short chat behind 413K, s (median of 4) | 0.508 | 0.511 | **0.494 (-3.0 %)** | **0.469 (-8.0 %)** | **0.501 (-1.6 %)** |
| first answer token, short chat behind 413K, s (thinking on; diagnostic) | 1.296 | 1.443 | **0.879** | **0.665** | **1.388** |
| C1 per-stream, tok/s | 547.32 | 540.42 | **559.46 (+2.9 %)** | **562.48 (+3.4 %)** | **547.51 (+0.7 %)** |
| C16 aggregate, tok/s | 2,453.8 | 2,379.4 | **2,403.1 (-0.6 %)** | **2,406.2 (-0.4 %)** | **2,421.9 (+0.2 %)** |
| cold prefill 32K, tok/s | 45,792.3 | 45,127.3 | **45,148.3 (-0.7 %)** | **45,657.5 (+0.4 %)** | **45,325.0 (-0.3 %)** |
| cold prefill 131K, tok/s | 43,213.2 | 43,366.8 | **43,335.0 (+0.1 %)** | **43,177.2 (-0.3 %)** | **43,257.8 (-0.1 %)** |
| cold prefill 300K, tok/s | 41,197.8 | 40,935.6 | **41,020.2 (-0.1 %)** | **41,006.8 (-0.1 %)** | **41,243.5 (+0.4 %)** |
| cold prefill 1M, tok/s | 28,129.2 | 28,035.7 | **28,062.1 (-0.1 %)** | **28,068.0 (-0.1 %)** | **28,377.4 (+1.1 %)** |
| C32 aggregate, single post-suite rep (mean of 9 categories), tok/s | 3,895.2 | 3,887.0 | **3,774.7 (-3.0 %)** | **3,659.5 (-6.0 %)** | **3,923.0 (+0.8 %)** |
| C32 `math` category, that rep, tok/s | 5,007.9 | 4,937.2 | **3,682.5** | **3,693.7** | **5,133.8** |
| qeval (75 tasks) | 71 | 71 | **71** | **71** | **71** |
| needle 3 @ 1M | 3/3 | 3/3 | **3/3** | **3/3** | **3/3** |
| greedy same-boot | 40/40 | 40/40 | **40/40** | **40/40** | **40/40** |
| JSON burst | 64/64 | 64/64 | **64/64** | **64/64** | **64/64** |
| DSML live | PASS | PASS | **PASS** | **PASS** | **PASS** |

Bars (CARD-X1-SPEC, cadence arms): decode during 413K ≥ 1.5× control mean, 413K time ≤ +6 %; decode during 1M ≥ 1.4×, 1M time ≤ +5 %; cold prefill ≥ −3 %; C1 ≥ −2 %, C16 ≥ −3 %, C32 ≥ −3 %; fidelity equal; true TTFT ≤ +10 %.

| bar | B (N=2) | C (N=3) | GR |
|---|---|---|---|
| decode during 413K ≥ 1.5× | PASS (1.83×) | PASS (2.72×) | n/a (prefill arm) |
| decode during 1M ≥ 1.4× | **FAIL** (1.35×) | PASS (2.84×) | n/a |
| C32 ≥ −3 % (single rep) | PASS (-3.0 %) | **FAIL** (-6.0 %) | PASS (+0.8 %) |
| GR: cold prefill ≥ +5 % at 131K or ≥ +8 % at 300K/1M | — | — | **FAIL** (131K -0.07 %, 300K +0.43 %, 1M +1.05 %) |

### Card X2 — long-only cadence (cadence2: N=3, MIN_REMAINING=8192; D1, D2) on Pg62 vs controls A1 / A2 / A3

| cell | gb300-pg62-A1 | gb300-pg62-A2 | gb300-pg62-A3 | **gb300-D1** | **gb300-D2** |
|---|--:|--:|--:|--:|--:|
| decode during 413K prefill, tok/s (trace-d median of 3) | 12.6 | 12.8 | 13.2 | **34.6 (2.69×)** | **32.2 (2.50×)** |
| decode during 1M prefill, tok/s (1 trace) | 10.2 | 11.1 | 10.4 | **28.6 (2.71×)** | **31.8 (3.01×)** |
| 413K prompt time while a 400-token reply streams, s (arrivals 2 / 4 / 6 s) | 11.39 / 11.15 / 11.00 | 11.21 / 11.15 / 11.01 | 11.29 / 11.13 / 10.93 | **12.18 / 11.71 / 11.41** | **12.17 / 11.81 / 11.44** |
| 413K prompt time while a 32-token probe runs, s (median of 4) | 10.85 | 10.91 | 10.89 | **10.96 (+0.7 %)** | **10.88 (-0.0 %)** |
| 1M prompt time while shared, s | 37.54 | 37.43 | 37.56 | **37.44 (-0.2 %)** | **37.23 (-0.7 %)** |
| true first token, short chat behind 413K, s (median of 4) | 0.508 | 0.511 | 0.543 | **0.371 (-28.7 %)** | **0.391 (-25.0 %)** |
| first answer token, short chat behind 413K, s (thinking on; diagnostic) | 1.296 | 1.443 | 1.494 | **0.730** | **0.758** |
| C1 per-stream, tok/s | 547.32 | 540.42 | 545.54 | **557.51 (+2.4 %)** | **549.23 (+0.9 %)** |
| C16 aggregate, tok/s | 2,453.8 | 2,379.4 | 2,386.3 | **2,404.5 (-0.1 %)** | **2,378.7 (-1.2 %)** |
| cold prefill 32K, tok/s | 45,792.3 | 45,127.3 | 44,968.0 | **44,791.2 (-1.1 %)** | **44,987.6 (-0.7 %)** |
| cold prefill 131K, tok/s | 43,213.2 | 43,366.8 | 43,627.0 | **43,176.9 (-0.5 %)** | **43,305.0 (-0.2 %)** |
| cold prefill 300K, tok/s | 41,197.8 | 40,935.6 | 40,717.7 | **41,121.7 (+0.4 %)** | **40,972.3 (+0.1 %)** |
| cold prefill 1M, tok/s | 28,129.2 | 28,035.7 | 28,164.3 | **28,112.6 (+0.0 %)** | **28,146.7 (+0.1 %)** |
| C32 aggregate, single post-suite rep (mean of 9 categories), tok/s | 3,895.2 | 3,887.0 | 4,001.5 | **3,794.8 (-3.4 %)** | **3,692.7 (-6.0 %)** |
| C32 `math` category, that rep, tok/s | 5,007.9 | 4,937.2 | 5,007.5 | **4,620.5** | **3,621.7** |
| qeval (75 tasks) | 71 | 71 | 71 | **70** | **71** |
| needle 3 @ 1M | 3/3 | 3/3 | 3/3 | **3/3** | **3/3** |
| greedy same-boot | 40/40 | 40/40 | 40/40 | **40/40** | **40/40** |
| JSON burst | 64/64 | 64/64 | 64/64 | **64/64** | **64/64** |
| DSML live | PASS | PASS | PASS | **PASS** | **PASS** |

| bar | gb300-D1 | gb300-D2 |
|---|---|---|
| decode during 413K ≥ 1.5× | PASS (2.69×) | PASS (2.50×) |
| 413K time ≤ +6 % (short-probe traces) | PASS (+0.7 %) | PASS (-0.0 %) |
| decode during 1M ≥ 1.4× | PASS (2.71×) | PASS (3.01×) |
| 1M time ≤ +5 % | PASS (-0.2 %) | PASS (-0.7 %) |
| cold prefill 32K/131K/300K/1M ≥ −3 % | PASS (-1.1 / -0.5 / +0.4 / +0.0 %) | PASS (-0.7 / -0.2 / +0.1 / +0.1 %) |
| C1 ≥ −2 % | PASS (+2.4 %) | PASS (+0.9 %) |
| C16 ≥ −3 % | PASS (-0.1 %) | PASS (-1.2 %) |
| C32 ≥ −3 % (single post-suite rep) | **FAIL** (-3.4 %) | **FAIL** (-6.0 %) |
| true TTFT behind 413K ≤ +10 % | PASS (-28.7 %) | PASS (-25.0 %) |
| fidelity: qeval ±1, needle 3/3, greedy 40/40, JSON, DSML | PASS (qeval 70, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) |

### Card X3 — cadence gated on a long prefill in flight (cadence3; E1, E2) on Pg62 vs controls A1 / A2 / A3 / A4

| cell | gb300-pg62-A1 | gb300-pg62-A2 | gb300-pg62-A3 | gb300-pg62-A4 | **gb300-E1** | **gb300-E2** |
|---|--:|--:|--:|--:|--:|--:|
| decode during 413K prefill, tok/s (trace-d median of 3) | 12.6 | 12.8 | 13.2 | 13.7 | **32.6 (2.49×)** | **33.8 (2.59×)** |
| decode during 1M prefill, tok/s (1 trace) | 10.2 | 11.1 | 10.4 | 10.4 | **28.2 (2.68×)** | **28.6 (2.72×)** |
| 413K prompt time while a 400-token reply streams, s (arrivals 2 / 4 / 6 s) | 11.39 / 11.15 / 11.00 | 11.21 / 11.15 / 11.01 | 11.29 / 11.13 / 10.93 | 11.38 / 11.13 / 11.02 | **12.13 / 11.65 / 11.40** | **12.14 / 11.69 / 11.46** |
| 413K prompt time while a 32-token probe runs, s (median of 4) | 10.85 | 10.91 | 10.89 | 10.92 | **10.87 (-0.2 %)** | **10.92 (+0.2 %)** |
| 1M prompt time while shared, s | 37.54 | 37.43 | 37.56 | 37.53 | **37.34 (-0.5 %)** | **37.72 (+0.5 %)** |
| true first token, short chat behind 413K, s (median of 4) | 0.508 | 0.511 | 0.543 | 0.460 | **0.371 (-26.5 %)** | **0.368 (-27.1 %)** |
| first answer token, short chat behind 413K, s (thinking on; diagnostic) | 1.296 | 1.443 | 1.494 | 1.392 | **0.669** | **0.756** |
| C1 per-stream, tok/s | 547.32 | 540.42 | 545.54 | 540.37 | **554.96 (+2.1 %)** | **551.72 (+1.5 %)** |
| C16 aggregate, tok/s | 2,453.8 | 2,379.4 | 2,386.3 | 2,374.7 | **2,407.0 (+0.4 %)** | **2,406.5 (+0.3 %)** |
| cold prefill 32K, tok/s | 45,792.3 | 45,127.3 | 44,968.0 | 45,102.8 | **45,301.2 (+0.1 %)** | **44,811.1 (-1.0 %)** |
| cold prefill 131K, tok/s | 43,213.2 | 43,366.8 | 43,627.0 | 43,193.1 | **43,380.3 (+0.1 %)** | **43,497.9 (+0.3 %)** |
| cold prefill 300K, tok/s | 41,197.8 | 40,935.6 | 40,717.7 | 41,145.0 | **41,244.4 (+0.6 %)** | **41,128.1 (+0.3 %)** |
| cold prefill 1M, tok/s | 28,129.2 | 28,035.7 | 28,164.3 | 28,112.1 | **28,042.2 (-0.2 %)** | **27,936.3 (-0.6 %)** |
| C32 aggregate, single post-suite rep (mean of 9 categories), tok/s | 3,895.2 | 3,887.0 | 4,001.5 | 3,850.6 | **3,824.2 (-2.2 %)** | **3,766.2 (-3.6 %)** |
| C32 `math` category, that rep, tok/s | 5,007.9 | 4,937.2 | 5,007.5 | 4,965.7 | **3,700.1** | **3,691.4** |
| qeval (75 tasks) | 71 | 71 | 71 | 71 | **71** | **71** |
| needle 3 @ 1M | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** | **3/3** |
| greedy same-boot | 40/40 | 40/40 | 40/40 | 40/40 | **40/40** | **40/40** |
| JSON burst | 64/64 | 64/64 | 64/64 | 64/64 | **64/64** | **64/64** |
| DSML live | PASS | PASS | PASS | PASS | **PASS** | **PASS** |

| bar | gb300-E1 | gb300-E2 |
|---|---|---|
| decode during 413K ≥ 1.5× | PASS (2.49×) | PASS (2.59×) |
| 413K time ≤ +6 % (short-probe traces) | PASS (-0.2 %) | PASS (+0.2 %) |
| decode during 1M ≥ 1.4× | PASS (2.68×) | PASS (2.72×) |
| 1M time ≤ +5 % | PASS (-0.5 %) | PASS (+0.5 %) |
| cold prefill 32K/131K/300K/1M ≥ −3 % | PASS (+0.1 / +0.1 / +0.6 / -0.2 %) | PASS (-1.0 / +0.3 / +0.3 / -0.6 %) |
| C1 ≥ −2 % | PASS (+2.1 %) | PASS (+1.5 %) |
| C16 ≥ −3 % | PASS (+0.4 %) | PASS (+0.3 %) |
| C32 ≥ −3 % (single post-suite rep) | PASS (-2.2 %) | **FAIL** (-3.6 %) |
| true TTFT behind 413K ≤ +10 % | PASS (-26.5 %) | PASS (-27.1 %) |
| fidelity: qeval ±1, needle 3/3, greedy 40/40, JSON, DSML | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) |
| bar 7: C32 per-category mean TTFT within +0.05 s of the control mean (gb300-E1) | **FAIL** (`math` 0.66 s vs 0.18 s, +0.48 s) |
| bar 7: C32 per-category mean TTFT within +0.05 s of the control mean (gb300-E2) | **FAIL** (`math` 0.64 s vs 0.18 s, +0.46 s) |

Single-rep C32 per-category mean TTFT, s (the instrument behind the X1–X3 C32 verdicts; the `math` and `summary` cells carry the stall):

| arm | coding | json | narrative | prose | math | reasoning | summary | format | ceiling_count |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gb300-pg62-A1 | 0.18 | 0.18 | 0.18 | 0.17 | 0.18 | 0.17 | 0.43 | 0.17 | 0.24 |
| gb300-pg62-A2 | 0.18 | 0.16 | 0.18 | 0.15 | 0.19 | 0.17 | 0.45 | 0.18 | 0.14 |
| gb300-pg62-A3 | 0.18 | 0.19 | 0.19 | 0.18 | 0.17 | 0.17 | 0.47 | 0.16 | 0.15 |
| gb300-pg62-A4 | 0.28 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.45 | 0.17 | 0.17 |
| gb300-B | 0.18 | 0.18 | 0.17 | 0.18 | 0.64 | 0.17 | 0.46 | 0.20 | 0.16 |
| gb300-C | 0.18 | 0.18 | 0.18 | 0.18 | 0.61 | 0.17 | 0.45 | 0.20 | 0.16 |
| gb300-D1 | 0.18 | 0.18 | 0.18 | 0.17 | 0.31 | 0.29 | 0.93 | 0.17 | 0.16 |
| gb300-D2 | 0.19 | 0.19 | 0.20 | 0.30 | 0.66 | 0.30 | 0.47 | 0.18 | 0.16 |
| gb300-E1 | 0.18 | 0.17 | 0.29 | 0.18 | 0.66 | 0.18 | 0.47 | 0.17 | 0.15 |
| gb300-E2 | 0.18 | 0.18 | 0.28 | 0.17 | 0.64 | 0.18 | 0.46 | 0.16 | 0.24 |

### Card X4 — cadence3 (G1, G2) vs same-window controls A6 (before) / A7 (after), on Pg62

| cell | gb300-pg62-A6 | gb300-pg62-A7 | **gb300-G1** | **gb300-G2** |
|---|--:|--:|--:|--:|
| decode during 413K prefill, tok/s (trace-d median of 3) | 12.2 | 12.5 | **33.3 (2.70×)** | **33.1 (2.68×)** |
| decode during 1M prefill, tok/s (1 trace) | 10.4 | 10.6 | **28.7 (2.73×)** | **30.4 (2.90×)** |
| 413K prompt time while a 400-token reply streams, s (arrivals 2 / 4 / 6 s) | 11.31 / 11.08 / 11.06 | 11.28 / 11.12 / 11.03 | **12.14 / 11.70 / 11.38** | **12.18 / 11.69 / 11.43** |
| 413K prompt time while a 32-token probe runs, s (median of 4) | 10.89 | 10.88 | **10.86 (-0.2 %)** | **10.86 (-0.2 %)** |
| 1M prompt time while shared, s | 37.47 | 37.53 | **37.04 (-1.2 %)** | **37.41 (-0.2 %)** |
| true first token, short chat behind 413K, s (median of 4) | 0.487 | 0.493 | **0.329 (-33.0 %)** | **0.362 (-26.1 %)** |
| first answer token, short chat behind 413K, s (thinking on; diagnostic) | 1.298 | 1.461 | **0.669** | **0.707** |
| C1 per-stream, tok/s | 541.25 | 546.13 | **543.76 (+0.0 %)** | **551.15 (+1.4 %)** |
| C16 aggregate, tok/s | 2,438.8 | 2,406.0 | **2,417.2 (-0.2 %)** | **2,397.1 (-1.0 %)** |
| cold prefill 32K, tok/s | 45,290.1 | 45,064.7 | **44,891.6 (-0.6 %)** | **45,648.6 (+1.0 %)** |
| cold prefill 131K, tok/s | 43,461.9 | 43,351.5 | **43,216.9 (-0.4 %)** | **43,607.7 (+0.5 %)** |
| cold prefill 300K, tok/s | 41,121.1 | 41,175.4 | **41,070.2 (-0.2 %)** | **40,954.6 (-0.5 %)** |
| cold prefill 1M, tok/s | 27,497.7 | 28,063.6 | **27,880.7 (+0.4 %)** | **28,056.7 (+1.0 %)** |
| C32 aggregate, single post-suite rep (mean of 9 categories), tok/s | 3,938.8 | 3,879.4 | **3,788.7 (-3.1 %)** | **3,913.4 (+0.1 %)** |
| C32 `math` category, that rep, tok/s | 4,716.5 | 4,485.3 | **3,600.0** | **4,840.5** |
| qeval (75 tasks) | 71 | 71 | **71** | **71** |
| needle 3 @ 1M | 3/3 | 3/3 | **3/3** | **3/3** |
| greedy same-boot | 40/40 | 40/40 | **40/40** | **40/40** |
| JSON burst | 64/64 | 64/64 | **64/64** | **64/64** |
| DSML live | PASS | PASS | **PASS** | **PASS** |

| bar | gb300-G1 | gb300-G2 |
|---|---|---|
| decode during 413K ≥ 1.5× | PASS (2.70×) | PASS (2.68×) |
| 413K time ≤ +6 % (short-probe traces) | PASS (-0.2 %) | PASS (-0.2 %) |
| decode during 1M ≥ 1.4× | PASS (2.73×) | PASS (2.90×) |
| 1M time ≤ +5 % | PASS (-1.2 %) | PASS (-0.2 %) |
| cold prefill 32K/131K/300K/1M ≥ −3 % | PASS (-0.6 / -0.4 / -0.2 / +0.4 %) | PASS (+1.0 / +0.5 / -0.5 / +1.0 %) |
| C1 ≥ −2 % | PASS (+0.0 %) | PASS (+1.4 %) |
| C16 ≥ −3 % | PASS (-0.2 %) | PASS (-1.0 %) |
| true TTFT behind 413K ≤ +10 % | PASS (-33.0 %) | PASS (-26.1 %) |
| fidelity: qeval ±1, needle 3/3, greedy 40/40, JSON, DSML | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) |
| C32×5 median ≥ −3 % vs median of the 10 control reps (G1) | PASS (3,905.6 vs 3,930.0, -0.62 %) |
| stall cells (>0.5 s mean TTFT) ≤ control max + 2 (G1) | PASS (0 vs controls 0 / 0) |
| C32×5 median ≥ −3 % vs median of the 10 control reps (G2) | PASS (3,928.6 vs 3,930.0, -0.04 %) |
| stall cells (>0.5 s mean TTFT) ≤ control max + 2 (G2) | PASS (0 vs controls 0 / 0) |

C32×5 per-rep means (tok/s, mean of 9 categories), warmed, back to back:

| arm | r1 | r2 | r3 | r4 | r5 | median |
|---|--:|--:|--:|--:|--:|--:|
| A6 | 3,845.3 | 4,011.4 | 3,973.8 | 4,017.0 | 3,917.9 | 3,973.8 |
| G1 | 3,905.6 | 3,865.1 | 3,804.9 | 3,967.0 | 3,917.9 | 3,905.6 |
| G2 | 3,817.5 | 3,962.6 | 3,949.5 | 3,928.6 | 3,849.2 | 3,928.6 |
| A7 | 3,859.4 | 3,942.0 | 3,891.3 | 3,953.7 | 3,823.3 | 3,891.3 |

### Cards X5 + X6 — prefill yield (Y1, Y2, Y3) vs same-window Pg63 controls K1 (before) / K2 (after)

| cell | gb300-pg63-K1 | gb300-pg63-K2 | **gb300-Y1** | **gb300-Y2** | **gb300-Y3** |
|---|--:|--:|--:|--:|--:|
| decode during 413K prefill, tok/s (trace-d median of 3) | 33.6 | 32.5 | **34.4 (1.04×)** | **32.2 (0.97×)** | **32.6 (0.99×)** |
| decode during 1M prefill, tok/s (1 trace) | 29.1 | 27.9 | **28.3 (0.99×)** | **28.2 (0.99×)** | **29.9 (1.05×)** |
| 413K prompt time while a 400-token reply streams, s (arrivals 2 / 4 / 6 s) | 12.01 / 11.72 / 11.35 | 12.15 / 11.71 / 11.45 | **12.21 / 11.83 / 11.40** | **12.16 / 11.71 / 11.51** | **12.16 / 11.77 / 11.38** |
| 413K prompt time while a 32-token probe runs, s (median of 4) | 10.88 | 10.91 | **10.89 (-0.1 %)** | **10.91 (+0.1 %)** | **10.91 (+0.2 %)** |
| 1M prompt time while shared, s | 37.84 | 37.97 | **37.56 (-0.9 %)** | **37.56 (-0.9 %)** | **37.53 (-1.0 %)** |
| true first token, short chat behind 413K, s (median of 4) | 0.418 | 0.352 | **0.211 (-45.2 %)** | **0.223 (-42.2 %)** | **0.185 (-51.9 %)** |
| first answer token, short chat behind 413K, s (thinking on; diagnostic) | 0.583 | 0.810 | **0.572** | **0.562** | **0.499** |
| C1 per-stream, tok/s | 547.63 | 552.16 | **547.07 (-0.5 %)** | **536.30 (-2.5 %)** | **545.81 (-0.7 %)** |
| C16 aggregate, tok/s | 2,387.6 | 2,399.9 | **2,393.6 (-0.0 %)** | **2,454.6 (+2.5 %)** | **2,401.5 (+0.3 %)** |
| cold prefill 32K, tok/s | 45,373.4 | 45,550.9 | **45,278.1 (-0.4 %)** | **45,685.0 (+0.5 %)** | **45,549.9 (+0.2 %)** |
| cold prefill 131K, tok/s | 42,937.9 | 43,122.3 | **43,059.3 (+0.1 %)** | **43,249.1 (+0.5 %)** | **43,708.7 (+1.6 %)** |
| cold prefill 300K, tok/s | 41,090.3 | 41,157.4 | **41,180.3 (+0.1 %)** | **41,053.4 (-0.2 %)** | **41,160.6 (+0.1 %)** |
| cold prefill 1M, tok/s | 27,959.9 | 28,172.9 | **28,117.7 (+0.2 %)** | **27,974.4 (-0.3 %)** | **28,052.0 (-0.1 %)** |
| C32 aggregate, single post-suite rep (mean of 9 categories), tok/s | 3,727.4 | 3,793.2 | **3,692.4 (-1.8 %)** | **3,788.5 (+0.8 %)** | **3,641.3 (-3.2 %)** |
| C32 `math` category, that rep, tok/s | 3,674.1 | 3,795.8 | **3,749.2** | **3,645.7** | **3,615.5** |
| qeval (75 tasks) | 71 | 71 | **71** | **71** | **71** |
| needle 3 @ 1M | 3/3 | 3/3 | **3/3** | **3/3** | **3/3** |
| greedy same-boot | 40/40 | 40/40 | **40/40** | **40/40** | **40/40** |
| JSON burst | 64/64 | 63/64 | **64/64** | **64/64** | **64/64** |
| DSML live | PASS | PASS | **PASS** | **PASS** | **PASS** |

| bar | gb300-Y1 | gb300-Y2 | gb300-Y3 |
|---|---|---|---|
| decode during 413K ≥ −5 % vs control (Pg63 base) | PASS (+4.1 %) | PASS (-2.6 %) | PASS (-1.4 %) |
| decode during 1M (no bar; reported) | 0.99× (-0.7 %) | 0.99× (-1.1 %) | 1.05× (+4.9 %) |
| cold prefill 32K/131K/300K/1M ≥ −3 % | PASS (-0.4 / +0.1 / +0.1 / +0.2 %) | PASS (+0.5 / +0.5 / -0.2 / -0.3 %) | PASS (+0.2 / +1.6 / +0.1 / -0.1 %) |
| C1 ≥ −2 % | PASS (-0.5 %) | **FAIL** (-2.5 %) | PASS (-0.7 %) |
| C16 ≥ −3 % | PASS (-0.0 %) | PASS (+2.5 %) | PASS (+0.3 %) |
| fidelity: qeval ±1, needle 3/3, greedy 40/40, JSON, DSML | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) | PASS (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML PASS) |
| P1: phase-locked short TTFT median ≤ 0.8× control median (Y1) | PASS (0.222 s vs 0.327 s → bar 0.261; 0.68×) |
| P2: 8,192-token prompt TTFT median ≤ 1.10× control (Y1) | PASS (1.107 s vs 1.069 s) |
| P1 long-prompt completion median ≤ +3 % (Y1) | PASS (10.806 s vs 10.789 s, +0.2 %) |
| C32×5 median ≥ −3 % vs the 10 K reps (Y1) | PASS (3,908.8 vs 3,956.4, -1.20 %) |
| stalls ≤ control max + 2 (Y1) | PASS (0; K1/K2 0 / 0) |
| P1: phase-locked short TTFT median ≤ 0.8× control median (Y2) | PASS (0.161 s vs 0.327 s → bar 0.261; 0.49×) |
| P2: 8,192-token prompt TTFT median ≤ 1.10× control (Y2) | PASS (1.102 s vs 1.069 s) |
| P1 long-prompt completion median ≤ +3 % (Y2) | PASS (10.820 s vs 10.789 s, +0.3 %) |
| C32×5 median ≥ −3 % vs the 10 K reps (Y2) | PASS (3,966.4 vs 3,956.4, +0.25 %) |
| stalls ≤ control max + 2 (Y2) | PASS (0; K1/K2 0 / 0) |
| P1: phase-locked short TTFT median ≤ 0.8× control median (Y3) | PASS (0.178 s vs 0.327 s → bar 0.261; 0.55×) |
| P2: 8,192-token prompt TTFT median ≤ 1.10× control (Y3) | PASS (1.119 s vs 1.069 s) |
| P1 long-prompt completion median ≤ +3 % (Y3) | PASS (10.870 s vs 10.789 s, +0.8 %) |
| C32×5 median ≥ −3 % vs the 10 K reps (Y3) | PASS (3,884.8 vs 3,956.4, -1.81 %) |
| stalls ≤ control max + 2 (Y3) | PASS (0; K1/K2 0 / 0) |

Phase-locked probe P1 (short chat, thinking off, sent at 0.8× that rep's solo cold-413K time; TTFT s, 5 reps) and P2 (same with an 8,192-token prompt, 2 reps):

| arm | P1 reps | P1 median | P2 reps | P2 median | long prompt done, s (median) | solo 413K, s (median) |
|---|---|--:|---|--:|--:|--:|
| K1 | 0.228 / 0.276 / 0.258 / 0.269 / 0.343 | 0.269 | 1.255 / 1.011 | 1.133 | 10.815 | 12.153 |
| Y1 | 0.279 / 0.136 / 0.222 / 0.181 / 0.276 | 0.222 | 1.083 / 1.130 | 1.107 | 10.806 | 12.127 |
| Y2 | 0.161 / 0.057 / 0.248 / 0.063 / 0.268 | 0.161 | 0.998 / 1.205 | 1.102 | 10.820 | 12.144 |
| K2 | 0.310 / 0.646 / 0.547 / 0.579 / 0.587 | 0.579 | 1.128 / 0.892 | 1.010 | 10.778 | 12.145 |
| Y3 | 0.256 / 0.176 / 0.282 / 0.094 / 0.178 | 0.178 | 1.165 / 1.072 | 1.119 | 10.870 | 12.147 |

C32×5 per-rep means (tok/s):

| arm | r1 | r2 | r3 | r4 | r5 | median |
|---|--:|--:|--:|--:|--:|--:|
| K1 | 3,885.7 | 3,939.2 | 3,919.2 | 3,926.8 | 3,856.7 | 3,919.2 |
| Y1 | 3,908.8 | 3,926.0 | 3,874.0 | 3,914.5 | 3,877.6 | 3,908.8 |
| Y2 | 3,889.4 | 3,969.5 | 3,995.4 | 3,914.4 | 3,966.4 | 3,966.4 |
| K2 | 3,994.0 | 3,980.0 | 3,973.6 | 4,041.9 | 3,994.8 | 3,994.0 |
| Y3 | 3,907.5 | 3,947.2 | 3,815.9 | 3,884.8 | 3,852.4 | 3,884.8 |

det_check (32 concurrent greedy requests with logprobs, run twice, 5,120 tokens compared): identical text+logprobs / 32 and logprob diffs —

| arm | K1 | Y1 | Y2 | K2 | Y3 |
|---|---|---|---|---|---|
| identical text / text+logprobs, of 32 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| logprob diffs (of 5,120) · max |Δ| | 4533 · 1.37 | 4435 · 1.43 | 4460 · 1.44 | 4425 · 1.44 | 4541 · 1.45 |

## Reading the tables

- **Pg63 (cards X1 → X4).** X1 found the effect (C: 2.72× / 2.84× decode beside 413K / 1M, fidelity identical) and the cost of deferring
  *every* prefill: C32 −6.0 % with `math` at 3,694 vs ~5,000 tok/s, because the short prompts of a 32-user burst were being deferred too. X2
  (defer only long prefills) and X3 (also tick only while a long prefill is in flight) kept the whole gain and still missed the single-rep C32
  bar on one or both boots, with the stall now visible as one category's 32-request TTFT at ~0.65 s instead of ~0.18 s. The X3b diagnostic
  showed that stall on Pg62 itself and on the inert overlay (code mounted, env off), and that the first C32 after any boot reads ~2,700: the
  instrument, not the overlay. X4 re-ran cadence3 with a warmed, repeated C32 (×5 per arm, controls included) and passed every bar on both
  boots: C32×5 medians −0.62 % / −0.04 % against the 10-rep control median, 0 stall cells on any arm. **The X1–X3 FAIL rows above are kept as
  recorded;** Pg63 was promoted on X4 alone.
- **The cadence's cost is on the long prompt, and only while a reply streams beside it.** `trace-d` `long_done` on the cadence arms reads
  12.13–12.18 / 11.65–11.87 / 11.38–11.46 s for a reply arriving at 2 / 4 / 6 s against 11.21–11.39 / 11.08–11.15 / 10.93–11.06 s on Pg62:
  in Card X4 (same window, G1/G2 vs the A6/A7 mean) **+7.5…+7.9 % / +5.3…+5.4 % / +3.0…+3.5 %** by arrival — the earlier the reply arrives,
  the more steps the prompt gives up; the Pg63/Pg64 boots later in the day read +6.3…+8.1 / +5.3…+6.6 / +2.7…+4.2 % against the same A6/A7
  mean (different window). With only a
  32-token probe beside it the long prompt moves −0.2 … +0.7 %; the solo cold-prefill cells move within ±1 %. The 1M prompt's shared time is
  −1.2 … +0.5 %. The brief's "+3…+8 %" is this row.
- **Pg64 (cards X5, X6).** Y1 and Y2 both passed every bar except Y2's C1 (−2.47 % against a −2 % bar; Y1 −0.5 %). Per the spec that is the
  "James-decides" band; James chose a third boot (X6) rather than a promotion or a drop. Y3 passed every bar (C1 −0.7 %), and Pg64 was promoted
  on Y1 + Y3 with Y2 on file. The Pg63 controls' own P1 medians were 0.269 s (K1) and 0.579 s (K2), so the control is pooled (0.327 s) and the
  per-boot spread is wide; C1 itself moves ±1.5 % boot to boot on this lane, which is how a −2 % bar gets missed by noise. The X5 "D" arms
  (b12x `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` on the sidecar) **never booted**: the sidecar died at preparation with
  `RuntimeError: planned dynamic direct routing is unsupported for this launch shape` on both attempts
  ([`promote/x5-D1-sidecar-boot-fail.log`](promote/x5-D1-sidecar-boot-fail.log)); closed, not a Pg64 component.
- **The yield does not change decode-beside-prefill** (Pg64's X5 bars for that row are "≥ −5 % vs Pg63", and read +4.1 / −2.6 / −1.4 %); the
  cadence does not change the first-token time of a short chat once the yield is on. They are additive and orthogonal, which is why the Pg64
  overlay carries both.
- **Fidelity on every arm, controls included:** qeval 71 (D1 read 70, inside the ±1 bar), needle 3/3 at 1M, greedy same-boot 40/40, JSON burst
  64/64 (K2 63/64), DSML live ALL_PASS. These are the card battery's checks, not the recipe's T2/T3 suite. det_check shows **0 of 32** concurrent
  greedy requests reproduce their logprobs on any arm, Pg63 controls included (first-token logprobs already differ): batch-composition
  nondeterminism is a property of this lane, and a scheduler overlay neither created nor removed it.

## Measured and not adopted (same window)

| lever | arm | result | disposition |
|---|---|---|---|
| vLLM #59131 (gather compressed KV once per KV-source group) + #63540 (fused inv-RoPE FP8 quant, 4 heads/program) as pure-Python overlays on v0.31.0 | X1 GR | engaged (`xfer #59131 retained span: hit 34,888 / miss 901` in the engine log); cold prefill 131K −0.07 %, 300K +0.43 %, 1M +1.05 % vs the A1/A2 mean — inside the control spread, below the +5 / +8 % bars | parked (overlays in the research dir: `ov/g59131`, `ov/r63540`; not shipped) |
| prefill cadence N=2 | X1 B | 1.83× beside 413K, 1.35× beside 1M (bar 1.4×), C32 −3.0 % | superseded by N=3 |
| cadence on every prefill (cadence, cadence2) | X1 C, X2 D1/D2 | the gain, with the 32-user burst stall | superseded by cadence3 (gated) |
| b12x deterministic combine on the sidecar | X5 D1, D2 | sidecar refuses to boot with our plans | closed; needs b12x work |
| per-position draft survival (W3, no-dark, live Pg63 09:53–10:01) | `raw/survival.json` | C1 T=0 survival by position 0.73 / 0.52 / 0.39 / 0.30 / 0.23 / 0.13 / 0.08; k=3 lanes (C ≥ 8) last row 0.43–0.46 | no card: dead-row skipping has no room on Pg63 |
| port of Owen's spark-ds41f engine to the GB300 | — | not attempted; his TileLang model loads only on capability family 120 and the block-scaled GEMMs / MXFP4 indexer need a tcgen05 rewrite | [why, public post](https://al-engr.com/gb300-why-not-port-spark-ds41f.html) |

## Known limits of this round

- **No soak, no T2/T3, no caller table, no rollback drill** on Pg63 or Pg64 (see the verdict block). The one restore that ran for real on
  2026-11-21 was the promotion script's own failure path at 09:15–09:22 (`xfer_arm.sh restore` → Pg62 serving), not `rollback-pg63-to-pg62.sh`.
  `rollback-pg63-to-pg62.sh` is byte-for-byte the 2026-11-17 `rollback-pg62y-to-pg62.sh` with the work-dir name changed, and that script
  **did** run end to end on 2026-11-17 01:23–01:30 CST (Pg62y → Pg62, rc 0, 7 min dark; [`promote/rollback-pg62y-20261117-0123.log`](promote/rollback-pg62y-20261117-0123.log)).
- **Long prompt pays while shared.** +3 … +8 % on a 413K prompt's completion while a reply streams beside it (by arrival); the 1M case was
  measured once per arm and reads −1.2 … +0.5 %. The reply's inter-token latency is bimodal (≈7 ms / ≈167 ms) by construction; a client p99
  will look worse than the mean.
- **Yield helps only single-chunk prompts.** An 8,192-token prompt behind a 413K prefill reads 1.10–1.12 s on Y1–Y3 vs 1.07 s on K1/K2 (P2).
  A 131K neighbour behind a 1M prefill is not helped by either lever (not measured this round; the 11/16 desk analysis).
- **One instrument shape.** `decode_during_prefill_tok_s` depends on how much of the reply overlaps the long prompt; compare within one
  trace shape. The brief's "33 vs 12.5 tok/s" is the `trace-d` median at arrivals 2/4/6 s.
- **Single-sample 1M cells** (one `trace-m1m` and one 1M prefill per arm).
- **Controls move.** Pg63 K1 vs K2 P1 medians 0.269 vs 0.579 s in the same hour; C1 ±1.5 % boot to boot. Bars at −2 % on C1 are inside that noise.
- **`prefill_capacity_bound` trap** (engine): stock sets it when blocked waiters exist, which disables throttling. Engagement was graded from the
  engine log on every arm (`deferred N of M steps` present on B, C, D1, D2, E1, E2, G1, G2, K1, K2, Y1, Y2, Y3; `none` on every Pg62 control).
- **Receipts withheld / compressed.** Per-request JSON (`trace-*.json`, `greedy-*.json`, `so_probe-*.json`, per-rep `bench-*.json`,
  `metrics-after-gates.txt`, engine diagnostic logs) is gzipped in place (`gzip -9 -n`, bytes preserved); LAN hosts and home paths are rewritten
  (the box's LAN address → `gb300`, the M4's → `m4`, both home directories → `~`). `battery-*.log` duplicates of `raw/battery/<arm>/battery.log` and empty
  driver stdout files were dropped. The `logs/post-*.log` files carry the greedy / JSON / DSML verdict lines.

## Files

| path | what |
|---|---|
| `cards/CARD-X1…X6-SPEC.md` | the six pre-registered specs, verbatim (scrubbed paths) |
| `cards/X1-table.md`, `X4-table.md`, `X5-table.md`, `ALL-table.md` | the orchestrator's per-card tables (`tools/table.py` output) |
| `patches/cadence3.diff`, `patches/yieldcad.diff`, `patches/SHA256SUMS` | the two overlays as unified diffs against stock v0.31.0 `scheduler.py`; hashes of the stock file and the served overlay files |
| `promote/promote-pg62-to-pg63.sh`, `promote-pg63-to-pg64.sh`, `rollback-pg63-to-pg62.sh`, `rollback-pg64-to-pg63.sh` | the box scripts (byte-identical to the box copies, paths scrubbed) |
| `promote/promote-pg63-20261121-0908.log` (false fail + auto-restore), `-0923.log` (Pg63), `promote-pg64-20261121-1201.log` (Pg64) | promotion logs |
| `promote/rollback-pg62y-to-pg62.sh`, `rollback-pg62y-20261117-0123.log` | the 2026-11-17 rollback the Pg63/Pg64 rollback scripts are cloned from, and its real run |
| `promote/lane-owner-20261121.txt` | every LANE-OWNER line of 2026-11-21 |
| `promote/x5-D1-sidecar-boot-fail.log` | the b12x deterministic-combine boot failure |
| `scripts/` | `driver*.py`, `measure.py`, `survival.py`, `diag_x3b.py`, `xfer_arm.sh` (Pg62-based arms, X1–X4), `xfer_arm2.sh` (Pg63-based arms, X5/X6; the copy here is the box's 14:53 revision whose `PROD=` already points at `pg64-work` for the later KL1 card — during X5/X6 it read `pg63-work`), `serve.sh.arm` (= `pg63-work/tree/serve/serve.sh`, sha256 `7ad4f7d9…`), `serve.sh.arm2` (= `pg64-work/tree/serve/serve.sh`, `8db3fe7e…`), `tools/` (mixed_trace, phase_trace, det_check, post_arm_gb300.sh, table.py) — box paths and LAN hosts scrubbed; not re-executed after scrubbing |
| `raw/battery/<arm>/` | the battery receipts per arm (phase JSONs, `runs.jsonl`, `battery.log`, raw bench/prefill/needle/mixed/qeval files) |
| `raw/post/<arm>/` | post-suite receipts per arm (traces, C32 log, gates, greedy, JSON burst, DSML) |
| `raw/x4-c32.json`, `raw/x5.json`, `raw/x6.json` | C32×5, det_check, P1/P2 and engagement per arm |
| `raw/c32x5/`, `raw/c32-single/` | every C32 bench file (X4/X5/X6 ×5 reps, the X3b diagnostic, the single post-suite reps) |
| `raw/logs/` | driver logs (every cell's timestamp and summary line), box op logs per arm, engine logs of the X3b diagnostic boots |
| `raw/survival.json` | W3 draft-survival cells on live Pg63 |

Credits: the prefill-yield mechanism is **Christopher Owen's** ([spark-ds41f](https://github.com/christopherowen/spark-ds41f), patch
`0545-Scheduler-Prefill-yield-a-long-prompt-s-aligned-chun.patch`, in his tree from commit
[`47342f26`](https://github.com/christopherowen/spark-ds41f/commit/47342f26f0c8fb530cfdcca98a666b12d58a3a9f), 2026-10-29); our part is the
port to stock v0.31.0 and the measurements. The cadence overlay reuses vLLM v0.31.0's own `prefill_schedule_interval` / `defer_prefills`
machinery (in v0.31.0 wired only for the DP engine core; single-engine since v0.32.1, vLLM #54627). The engine under both is **Garner McCloud's** (credited in the recipe README).
