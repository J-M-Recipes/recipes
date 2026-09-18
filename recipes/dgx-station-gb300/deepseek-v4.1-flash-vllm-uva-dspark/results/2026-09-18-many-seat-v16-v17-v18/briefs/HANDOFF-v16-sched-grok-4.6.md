# HANDOFF — v16-sched: scheduler tuning for 16–24 seats (fund workload) — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. Read fully before acting. All rules in `HANDOFF-grok-4.6.md` apply: never `:30003`, never start GLM (`glm53-*`), never `docker rm`, never `docker pull`, drop caches before a fresh launch (`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`), fast-fail grep only `ValueError:|Traceback|ERROR|RuntimeError:`, stop means stop, findings only, no invented numbers. Ledger entries in `results/ledger.md` per milestone (name model + provider: `grok-4.6 · xai-oauth`). Box: `ssh <user>@<station>`. Project on the Mac: `$HOME/hermes/pin-hot-experts/` (you run from here; write results here).

## Why

The fund-workload post (https://al-engr.com/gb300-35-seat-fund-workload.html, re-measured on v15 2026-09-18 05:20 CDT, receipts `~/iwyzer/iw-dsv41-v15-reconfirm.jsonl` + `run_dsv41-v15-reconfirm.log` on the box) shows DeepSeek-V4.1 at C16: warm agent turn TTFT p95 **8.54 s** (bar ≤ 2), chat (10.7K prompt) TTFT p95 **28 s**, cold 120K under 16 streams p95 **15.9 s** (bar ≤ 15), and at C24 warm p50 **31 s**.

Milo measured 2026-09-18 on the idle v15 lane: cold prefill is **fine** — 22K tok/s at 26K–104K (4.7 s TTFT at 104K), 20.5K at 207K, 18.5K at 405K, and 21K tok/s for a 52K prompt with 8 decode streams live. **Raw prefill did not regress under pin-hot-experts.** The container logs from the fund run show the real cause:

- `Running: 16` pinned at `--max-num-seqs 16`, `Waiting: 1` at C16, `Waiting: 10–12` at C24 → the C24 numbers are a slot queue. **KV usage peaked at 16%** (2.33M tokens of KV, 0.9–1.3M needed).
- Budget starvation. In this build's V1 scheduler (`v1/core/sched/scheduler.py:598`) `running` is served FCFS and each request's chunk is capped at `--long-prefill-token-threshold` (6144) per step. With the fund mix (25% 90K docs, 35% 46K agent prompts) two prefills in flight take 6144 + 2048 = the entire 8192 budget, and every later request — a 3K warm turn, a 10K chat prompt — gets **zero** tokens per step until one of them finishes (a 90K doc ≈ 15 steps ≈ 7–10 s under load). That is the 8.5 s / 28 s p95. (`max_num_partial_prefills` / `max_long_partial_prefills` do **not** exist in this build — verified against `EngineArgs`; do not add them.)

This run tests **scheduler knobs only**. The hook, rowmap, offload, image, util and batched-tokens are untouched.

## One axis: scheduler

`scripts/launch-v16-sched.sh` (Mac; copy to box `$BOX_HOME/pin-hot-experts/scripts/`) = v15 launch with:

| flag | v15 | v16 |
|---|---|---|
| `--max-num-seqs` | 16 | **32** |
| `--long-prefill-token-threshold` | 6144 | **2048** (≥4 prefills share each 8192 step; a 3K turn lands in 2 steps) |
| k-schedule | `[[1,4,5],[5,16,1]]` | **`[[1,4,5],[5,32,1]]`** |

Everything else identical (`PIN_ROWMAP=/w/rowmap-static-v1.json`, off60, util 0.97, batched 8192, 1M ctx). Leave `--max-num-batched-tokens` at 8192 — raising it grows the ~4.5 GiB activation peak against only 4.75 GiB of KV; not this run.

**Autotune:** `max-num-seqs` changes CUDA-graph capture sizes and may produce a new autotune hash → up to ~75 min silent GPU-100% tune before bind. That is expected, not a failure. Confirm bind via `Loaded N configs` in the log and note hash hit/miss + tune minutes in the ledger. If it OOMs at graph capture or KV sizing (32 seqs × DSpark graphs), fall back to `SEQS=24 KSCHED='[[1,4,5],[5,24,1]]'` and record both attempts.

**Known cost of lpt 2048:** a lone cold doc now prefills in 2048-token chunks (idle 104K ≈ 51 steps instead of 17); expect single-stream cold TTFT to lose some percent. Measure it: after bind and before the windows, run `BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none SIZES="8000 32000 128000" N=2 THINKING=0 python3 $BOX_HOME/dsv41/cold_prefill_probe.py` on v16 (Milo's v15 idle reference: 16.8K / 22.3K / 22.0K tok/s, 4.72 s at 103.6K). **One permitted follow-up:** if the warm-turn bar passes but the cold-120K-under-load bar or the idle probe loses >25%, reboot with `LPT=4096` (scheduler-only change → same autotune hash, ~5-min bind) as `dsv41-vllm-v16b-sched-lpt4096-EXP` and run one more v16b→v15→v16b cycle. Nothing else changes.

**Verify the hook fired:** the log must show `PIN_HOT rowmap layer … hot=295 cold=89` and the re-home report (HBM_expert ≈ 206.6 GiB, pinned_expert ≈ 62.3 GiB). A boot without those lines is plain positional offload — stop and report.

## Procedure

### 0. Box state
`docker ps` must show only `dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` (or nothing). GLM stopped-and-kept; `:30003` down. Record `nvidia-smi` GPU1 used/total and `MemAvailable`.

### 1. Candidate boot
Stop-and-keep v15 (`docker stop`, no rename). Drop caches. Run `launch-v16-sched.sh`. Wait for `/v1/models` on `:30006`. Record: hash hit/miss, tune time, `GPU KV cache size` line, graph-capture memory line, hook re-home lines, bind time.

### 2. Same-window measurement: v16 → v15 → v16
Each window, in this order, from the box `$BOX_HOME/iwyzer/` (the fund harness; `CHARS_PER_TOKEN=3.894`, `BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva`):
```
TAG=<tag>-warm python3 iwbench.py mix 8 120        # warm pass, discarded
TAG=<tag>-C16  python3 iwbench.py mix 16 420
TAG=<tag>-C24  python3 iwbench.py mix 24 420
TAG=<tag>-cold python3 iwbench.py coldload 5
TAG=<tag>-tools python3 iwbench.py tools 16
```
then `knee.sh` ×2 from `$BOX_HOME/dsv41/` (C1/C8/C16 knee; the decode-cost check). Tags: `v16a`, `v15ctl`, `v16b`; `OUT=iw-<tag>.jsonl`, logs `run-<tag>.log`. Model `v15ctl` = `docker start` the `-BOUND-REF` (hash-hit, ~5 min bind). Stop-and-keep between windows. Do not run anything else against `:30006` during a window. Copy all jsonl + logs back to `results/v16/`.

### 3. Leave up
If **all** bars pass: leave v16 UP on `:30006`, v15 stopped-and-kept (do NOT rename either; Milo promotes). Otherwise `docker start` v15 and stop-and-keep v16 renamed with a verdict suffix (`-EXP-<reason>-FAIL`), never rm.

## Win bars (v16 two-window means vs v15ctl)
- C16 warm agent turn TTFT p95: **≤ 2.0 s** (target) — hard bar: ≤ 4.0 s (≥ −50%)
- C16 chat TTFT p95: ≤ 10 s (from 28)
- Cold 120K under 16 streams TTFT p95: ≤ 15 s
- C24 warm agent turn p50: single digits (from 31 s)
- C16 prose decode p10: ≥ −10% vs v15 (accept a small decode cost for the latency win; report the number)
- `knee.sh` C1: ≥ −1.5%; C8/C16: ≥ −5%
- Tools 64/64
- Boot fits: KV ≥ 2.0M tokens (log line); host Δ ≤ 2 GiB
- No `ERROR|Traceback` in the serving log during windows

## Outputs
`results/v16/{README.md, iw-*.jsonl, run-*.log, knee-*.json, boot-log-excerpt.txt}`, `scripts/launch-v16-sched.sh` (final values used), ledger entries.
Verdict line: `V16 VERDICT: warm p95 C16 … s (v15 …) · chat p95 … · cold120K p95 … · C24 warm p50 … · prose p10 C16 Δ…% · knee C1 Δ…% · tools …/64 · hash hit yes/no (… min) · KV … M tok · left up: v16|v15`. ≤ 15-line final summary, then stop.
