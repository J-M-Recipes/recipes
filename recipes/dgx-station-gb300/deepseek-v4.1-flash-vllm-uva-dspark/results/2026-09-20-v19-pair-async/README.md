# 2026-09-20 — v19 candidate second pair + fund harness · `--async-scheduling` at v18 — DSV4.1-Flash / GB300 `:30006`

Runner: `v19-async-runner-2026-09-20.sh` (Milo, nohup on the box, 08:00–09:48 CDT). Verdicts read from receipts by Milo (Hermes `milo` profile, claude-fable-5-1 via anthropic). Reference throughout: `dsv41-vllm-v18-cgsizes-BOUND-REF` (0909 image, v15 hook, off60, seqs 24, `[[1,4,5],[5,24,1]]`, token capture sizes, fp8 KV). v18 is live on `:30006` at the end; **nothing promoted**.

Two questions, one box session, every verdict same-window:

- **A. v19 candidate on the LOADED configuration** — nightly `dee37d89` + v15 hook + `--cpu-offload-gb 54` + `--kv-cache-dtype fp8_ds_mla` + the `ddf01704…` FlashInfer autotune set already in the cache dir (T3b6 saved it; T3b6b loaded it once at 183.3). Was 183 a stable draw, and what does the lane's actual job (the 35-seat fund harness, `iwbench.py`) say? Windows **v19a → v18ctl9 → v19b**, each = `knee.sh` ×2 + `iwbench` mix 8/120 (warm, discarded) · mix 16/420 · mix 24/420 · coldload 5 · tools 16.
- **B. `--async-scheduling` on the v18 launch** (0909 image, hook, every v18 flag + one). Was a wash at v12 (11 ms step); re-asked at v18's ~5.8 ms step after Cursor's MoK post reported Grace-side CPU work catching the GPU on GB300. Windows **ASYNC → v18ctl10**, each = `run_window_T1T2.sh` (knee ×2, knee6, agent fixture, replay n=4 ×2).

## Boot facts

| boot | bind | autotune | KV | residency |
|---|---|---|---|---|
| v19a 08:00 | 480 s | `ddf01704…` **Loaded 189**, 0 new | fp8 · 3,179,961 tok (3.03×) | hook rehome 40 layers, HBM_expert 206.61 / pinned 62.33 GiB (= v18) |
| v19b 08:55 | 480 s | `ddf01704…` Loaded 189, 0 new | fp8 · 3,179,961 | same |
| ASYNC 09:24 | 470 s | v18 hash **Loaded 189** (hash hit — `--async-scheduling` does not enter the kernel-shape hash) | fp8 · 2,502,950 (= v18) | same |
| v18ctl9 / v18ctl10 | 300 / 290 s (`docker start`) | — | 2.50M | — |

`launch-many-seat.sh` gained an `EXTRA` passthrough for B (`.bak-pre-extra` kept; empty default = v18 launch byte-identical). Smoke 5/5 on all three candidates.

## A. v19 — knee + fund harness, same window

| | C1 (r1 / r2) | C8 | C16 (r2) | fund C16 agg | C16 warm agent p50 / **p95** | fund C24 agg | C24 warm agent p50 / p95 | cold 120K/16 p95 | tools |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| v18ctl9 | 171.6 / 172.0 | 660–664 | 957 | 310 | 0.534 / **1.778** | 285 | 0.602 / 2.264 | 8.44 | 64/64 |
| **v19a** | **183.2 / 183.2** | 629 | 986 | 308 | 0.532 / **1.684** | **306** | 0.557 / 2.196 | 8.33 | 64/64 |
| **v19b** | **183.1 / 183.1** | 628–629 | 987 | 306 | 0.547 / **2.558** | **310** | 0.545 / 2.283 | 8.32 | 64/64 |

Fund-harness detail (chat / doc TTFT p50, C16): v18ctl9 0.88 / 6.71 s · v19a 0.83 / 6.36 · v19b 1.06 / 6.58. C24 decode p50 per stream: 13.9 / 15.1 / 14.6 tok/s. Power under the fund mix: all three windows 455–640 W, cap 1300 W, throttle `0x0`.

**Parity:** v19a vs v19b **18/18** — the two loaded boots are token-identical on every prompt including the agent-tool prompt that jitters on every other pair we have. (Hook-on live-vs-loaded pairs were 16–17/18; controls vs each other 17/18.) v19 vs v18ctl9 2/18 — cross-image, not a bar.

### Bars (runner header) and result

| bar | v19a | v19b | verdict |
|---|---|---|---|
| C1 ≥ 172 × 1.05 = 180.6, both boots | 183.2 | 183.1 | **PASS** (+6.6%, four runs within 0.1) |
| fund C24 warm p50 ≤ control (0.602) | 0.557 | 0.545 | **PASS** |
| fund C16 warm p95 ≤ control + 0.5 (2.278) | 1.684 | **2.558** | **v19b MISS** by 0.28 s (one p95 on n=71; v19a passes by 0.6 s) |
| tools 64/64 | 64/64 | 64/64 | PASS |
| (not a bar, recorded) C8 knee | −5% | −5% | the T3b6/T3b6b C8 dip reproduces on the loaded config: 629 vs 660 |
| (not a bar, James's call) fixture tool_json | — | — | T3b6: −6% vs −3% bar; not re-measured here |

**Reading.** 183 is not a lottery draw: four C1 runs on two fresh boots land within 0.1 tok/s, and the two boots produce identical tokens. On the lane's real job the candidate is at or better than v18 on C24 aggregate (+8%), C24 warm p50, cold-load p95 and tools; C16 aggregate is a wash. The single miss is v19b's C16 warm-agent p95 (2.56 s), which v19a beat by 0.6 s in the same session — a third C16 window (15 min on a hash hit) would settle it. C8 knee is −5% on every loaded nightly boot, unexplained and recorded. **VERDICT A: v19 candidate HOLDS; not promoted.** Open before promotion: one more C16 fund window for the p95, and James's ruling on tool_json −6% for a tool-heavy lane.

## B. `--async-scheduling` at v18

| | C1 (r1 / r2 / k6) | C8 | C16 (r2 / k6) | replay r1 / r2 | fixture prose / tool_json / weighted accept |
|---|--:|--:|--:|--:|---|
| **ASYNC** | 170.8 / 171.1 / 171.3 | 651–657 | 945 / 939 | 337 / 450 | 190.6 / 240.9 · 0.628 |
| v18ctl10 | 171.7 / 171.9 / 172.1 | 655–657 | 947 / 950 | 334 / 455 | 202.4 / 227.5 · 0.628 |

C1 −0.4%, C8 −0.3%, C16 −0.6%, replay ±1%, acceptance identical, parity 17/18 (the usual tool prompt). Bar was C1 and C8 ≥ +2%. **VERDICT B: `--async-scheduling` CLOSED at v18** as at v12. At a 5.8 ms decode step the Grace-side scheduler is still not on the critical path of this lane; the MoK observation is a training-loop finding and does not transfer here. Do not re-spend a boot on it.

## Box state at end (09:48 CDT)

`:30006` = `dsv41-vllm-v18-cgsizes-BOUND-REF`, 207 W idle. Stopped-and-kept: `dsv41-vllm-v19a-…-ddf01704-EXP`, `dsv41-vllm-v19b-…-ddf01704-EXP`, `dsv41-vllm-ASYNC-v18-async-scheduling-EXP`. Nothing removed; no cache dirs moved. Receipts: `receipts/` (knee/agentfix/replay JSON, `iw-*.jsonl` fund captures, `run-*.log`, boot/smoke excerpts, power lines, `parity/`). Runner bug carried from T3c: the receipt `mv` glob does not cover `run_window_T1T2.sh`'s output dir; files were moved by hand. Fix the template before the next runner.
