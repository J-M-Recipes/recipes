# 2026-10-26 — cosine gate R2 re-judge: v25m **R2 PASS** (R1 still FAIL, reported)

The rule was registered on 2026-10-24 at ~08:15 CDT, before any log was re-judged under it
([`REGISTERED.md`](REGISTERED.md)). The re-judge ran on 2026-10-26, starting 17:06:47 CDT, on the Station's CPU. It used
the unmodified box `judge_r2.py` (sha1 `b2b9d66fbbe0…`, the registered sha1) with the argument list that
[`cardGS_runner.sh`](cardGS_runner.sh) builds. Nothing booted, and no GPU was used. The archived server logs were read
read-only.

## The rule (verbatim from `REGISTERED.md`)

> **R2.** Same scope as R1. For each check, from the fields `mega_peer_hook._log_check` already prints:
> `e = rel_diff × cold_share = ||cold_peer − cold_trt|| / ||y_trt||`, the sidecar's error as a fraction of the whole layer output.
> - coverage: all 40 layers have a >= 32-row check (unchanged)
> - 0 peer + 0 hook timeouts (unchanged)
> - **bar B = 1.5 × the largest per-boot max e over the reference boots**, all on rowmap-static-v1 (v25 static):
>   SC0 arm A (2026-10-21) + Card GS K1 and K2 (2026-10-26, after the v25m 48 h soak)
> - candidate PASS iff coverage, timeouts, and max e <= B
> - reported, not gated: R1 verdict, min cos over checks with cold share >= 1 %, p99 e

## Per-boot table (`judge_r2.py` output, verbatim; `R2-VERDICT.md`, `judge.stdout`, `r2.json`)

Bar B = 1.5 × max reference e = **0.04646** (1 v25-static reference boot).

| boot | kind | gated checks | layers | max e (err / layer output) | p99 e | worst-e check | R2 | R1 (old) min cos | min cos, share >= 1% |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| v25-static-SC0 | ref | 1341 | 40 | 0.03098 | 0.01076 | L35 T=128 rows=92 cos 0.99949 share 0.885 | **PASS** | 0.99906 (PASS) | 0.99921 |
| 285-99-static-h285-SC3b | cand | 1251 | 40 | 0.01740 | 0.01079 | L38 T=128 rows=116 cos 0.99981 share 0.883 | **PASS** | 0.99885 (FAIL) | 0.99939 |
| mix-v1-RonanLabs-SC3 | cand | 1179 | 40 | 0.01506 | 0.00873 | L10 T=2783 rows=1508 cos 0.99988 share 0.953 | **PASS** | 0.99911 (PASS) | 0.99911 |
| v25m-GV | cand | 1439 | 40 | 0.01487 | 0.01158 | L38 T=128 rows=83 cos 0.99979 share 0.729 | **PASS** | 0.99811 (FAIL) | 0.99811 |
| v25m2-L39static-GV2 | cand | 1428 | 40 | 0.01538 | 0.01167 | L36 T=128 rows=90 cos 0.99969 share 0.608 | **PASS** | 0.99831 (FAIL) | 0.99928 |

All boots had 0 hook timeouts and 0 peer timeouts (`r2.json`).

## Verdict

**v25m (rowmap-mixed-ar-v1.json, Card GV K boot): R2 PASS.** Coverage was 40/40 layers, with 0 hook and 0 peer
timeouts over 1,439 gated checks. Max e = **0.01487** (L38 T=128 rows=83, cos 0.99979, cold share 0.729), against
B = **0.04646**. p99 e was 0.01158 (v25 static SC0: 0.01076).

**R1 (reported, not gated) still FAILS.** The min cos is 0.99811 at L39 T=128 rows=49 (cold share 0.079, e = 0.0049).
That is 1 of 1,439 checks below 0.999. The other candidates also pass R2: h285 0.01740 (R1 FAIL), mix-v1 0.01506 and
v25m2 0.01538 (R1 FAIL).

v27 runs on the same map (v27 = v25m-kvtier + the block-verification key). The checks judged here come from the
Card GV check-mode boot of v25m (2026-10-24), not from a v27 check-mode boot; none has been run.

## Caveats (published verbatim from the re-judge)

- **Deviation from the registration.** B was computed from 1 of the 3 registered reference boots (SC0 arm A only). The
  registered set was SC0 + Card GS K1/K2. The K1/K2 boots (two fresh rowmap-static-v1 check-mode boots) never ran,
  because Card GS was cancelled on 2026-10-24 (`results-cardGS/runner.log` stops at "ARMED"). Running them needs GPU
  lane boots, which this re-judge was not allowed to make. The runner adds K1/K2 only if `serverlog-K1/K2.log` is
  non-empty, so the unmodified script ran with SC0 alone.
- **The verdict cannot flip.** B = 1.5 × the max over the references, so adding references can only raise B. The
  coverage and timeout terms depend only on the candidate. v25m's max e (0.01487) is already below SC0's own max e
  (0.03098), so it would pass even at 1.0× a single reference. The value of B will change once K1/K2 exist; the v25m
  verdict will not.
- **B rests on one check.** SC0's max e comes from one outlier check (L35, e 0.03098); the next highest is 0.01502.
  v25m's max e is also below that second-highest SC0 e.
- **R2 is a numerics gate only.** It does not replace the 48 h soak, which was cancelled and never cleared.

## Files

| file | what |
|---|---|
| `REGISTERED.md`, `judge_r2.py`, `cardGS_runner.sh` | the registration and its scripts (already in this dir before the run; `judge_r2.py` here has its paths scrubbed, so its sha1 differs from the box copy `b2b9d66fbbe0`) |
| `R2-VERDICT.txt` | the full verdict: rule, what ran, input sha1s, boot facts, per-boot table, per-check detail, deviation |
| `R2-VERDICT.md`, `judge.stdout`, `r2.json` | `judge_r2.py`'s own outputs (table + machine-readable per-boot numbers) |
| `inputs-sha1.txt`, `boot-facts.txt` | sha1 of the script, runner, 7 archived logs and 2 rowmaps; each log's map and check-mode line |
| `perchecks-*.csv`, `perchecks.stdout`, `r2_perchecks.py` | a report-only helper. It imports `judge_r2.RX` unmodified and dumps every gated check (L, T, rows, rel, cos, share, e) for each boot |
| `start.txt`, `end.txt` | box `date` around the judge (both 17:06:47 CDT; the judge runs in well under a second) |
| `SHA1SUMS-box-originals.txt` | sha1 of the receipts as written on the box, before the path scrub (`R2-VERDICT.txt` = `5976e341…`) |
| `SHA1SUMS-published.txt` | sha1 of the files as published here (after the scrub) |

Scrub: the box home directory → `~` in `R2-VERDICT.txt`, `r2.json`, `inputs-sha1.txt` and `boot-facts.txt` (37 replacements).
The sha1s listed in `inputs-sha1.txt` are of the original box files, which were not changed.
