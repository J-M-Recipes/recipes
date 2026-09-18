# HANDOFF — v17-seqs24: isolate the max-num-seqs axis (fund workload, C24 queue) — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. Read fully before acting. All rules in `HANDOFF-grok-4.6.md` apply: never `:30003`, never start GLM (`glm53-*`), never `docker rm`, never `docker pull`, drop caches before a fresh launch, fast-fail grep only `ValueError:|Traceback|ERROR|RuntimeError:`, stop means stop, findings only, no invented numbers. Ledger entries in `results/ledger.md` per milestone (`grok-4.6 · xai-oauth`). Box `ssh <user>@<station>`. Project `$HOME/hermes/pin-hot-experts/`. **Read `results/v16/README.md` first** — v17 exists because of it.

## Why

v16 (seqs 32 + lpt 2048) proved the C24 slot queue is real and fixable — warm-turn p50 **27.1 → 3.6 s** same-window, knee C1/C16 +7% — but bundled two knobs. `lpt 2048` cost idle prefill −47% and C16 prose p10 −29%; 32-seq DSpark graphs cost 2.85 GiB and cut KV 2.26M → 1.39M tokens. v17 keeps **only** the seqs axis, at the fund's stress case (24), with everything else back at v15.

## One axis: `--max-num-seqs` 16 → 24

`scripts/launch-v16-sched.sh` with env `NAME=dsv41-vllm-v17-seqs24-EXP SEQS=24 LPT=6144 KSCHED='[[1,4,5],[5,24,1]]'` (the script's DUMP dir is `v16`; that is fine). Everything else identical to v15 (`PIN_ROWMAP=/w/rowmap-static-v1.json`, off60, util 0.97, batched 8192, lpt **6144**, 1M ctx). Verify in `docker inspect` Args after launch: `--max-num-seqs 24`, `--long-prefill-token-threshold 6144`, ksched `[[1,4,5],[5,24,1]]`.

Expect an **autotune miss** (v16's was ~16 min; hash `b8c6fdcc` was the 32-seq shape, 24 will be new). Record hash + minutes. Hook lines required: `hot=295 cold=89`, `HBM_expert≈206.61 GiB`, `pinned_expert≈62.33 GiB`; absent → stop and report.

**KV gate (read at bind, before any window):** the `GPU KV cache size` line. Bar **≥ 1.8M tokens**. If below 1.8M, do NOT run windows yet: stop-and-keep the container, and boot a second attempt `NAME=dsv41-vllm-v17b-seqs24-cg-EXP` with the same env plus `--cudagraph-capture-sizes 1 2 4 8 12 16 24` appended (that is the flag on this build — `--cuda-graph-sizes` does not exist; verified in `EngineArgs`) (edit a copy of the launcher `scripts/launch-v17b-cg.sh`; do not modify `launch-v16-sched.sh`). Record graph-capture GiB and KV for both. Proceed to windows with whichever boots ≥ 1.8M; if neither, report and restore v15.

## Procedure

0. Box state: only `dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` running (or nothing). GLM stopped-and-kept; `:30003` down. Record `nvidia-smi` GPU1 + `MemAvailable`.
1. Stop-and-keep v15. Drop caches. Launch. Wait for `/v1/models` on `:30006`. Record hash hit/miss, tune minutes, graph GiB, KV tokens, hook lines, bind time.
2. Idle probe on v17: `cd $BOX_HOME/dsv41 && BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none SIZES="8000 32000 128000" N=2 THINKING=0 python3 cold_prefill_probe.py`. Reference (v15 idle, Milo 09-18): 16.8K / 22.3K / 22.0K tok/s, 4.72 s at 103.6K.
3. Same-window **v17a → v15ctl → v17b**, per window in this order from `$BOX_HOME/iwyzer/` (`BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none CHARS_PER_TOKEN=3.894 THINKING=0`):
   `mix 8 120` (discard) · `mix 16 420` · `mix 24 420` · `coldload 5` · `tools 16` · then `knee.sh` ×2 from `$BOX_HOME/dsv41/`. Tags `v17a`, `v15ctl2`, `v17b`; `OUT=iw-<tag>.jsonl`, logs `run-<tag>.log`. v15ctl2 = `docker start` the `-BOUND-REF` (hash-hit). Stop-and-keep between windows; nothing else on `:30006` during a window. Copy all jsonl/logs/knee json to `results/v17/`.
4. Leave-up: all bars pass → v17 UP, v15 stopped-and-kept, **no renames** (Milo promotes). Otherwise `docker start` v15, rename v17 `-EXP-<reason>-FAIL`, never rm.

## Win bars (v17 two-window means vs v15ctl2)
- C24 warm agent turn TTFT p50: **≤ 5 s** (v15 was 27.1 s) — the point of the run
- C24 prose decode p10: report; no bar (v15 had a queue, comparison is apples/oranges)
- C16 warm agent TTFT p95: ≤ v15ctl2 + 0.5 s
- C16 chat TTFT p95: ≤ v15ctl2 × 1.1
- C16 prose decode p10: **≥ −5%** vs v15ctl2 (lpt is back at 6144, so the v16 −29% should not recur; if it does, that is a seqs finding — record it)
- Cold 120K under 16 streams p95: ≤ v15ctl2 × 1.1
- Idle cold prefill 128K: ≥ −5% vs 22.0K tok/s
- knee C1 ≥ −1.5%; C8/C16 ≥ −5%
- Tools: ≥ v15ctl2's count (both configs jitter at 61–64; report both)
- KV ≥ 1.8M tokens; host Δ ≤ 2 GiB; no `ERROR|Traceback` in serving log

## Outputs
`results/v17/{README.md, iw-*.jsonl, run-*.log, knee-*.json, cold_prefill_probe.out, boot-log-excerpt.txt}`, `scripts/launch-v17b-cg.sh` if used, ledger entries.
Verdict line: `V17 VERDICT: C24 warm p50 … s (v15 …) · C16 warm p95 … (v15 …) · chat p95 … · cold120K p95 … · prose p10 C16 Δ…% · idle 128K … tok/s · knee C1 Δ…% · tools …/64 vs …/64 · hash hit no (… min) · graphs … GiB · KV … M tok · left up: v17|v15`. ≤ 15-line final summary, then stop.
