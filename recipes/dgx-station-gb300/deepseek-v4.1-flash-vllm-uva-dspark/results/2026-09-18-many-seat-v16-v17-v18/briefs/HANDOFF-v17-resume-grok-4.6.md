# HANDOFF — v17 windows (resume) — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. Read fully before acting. All rules in `HANDOFF-grok-4.6.md` apply: never `:30003`, never start GLM (`glm53-*`), never `docker rm`, never `docker pull`, fast-fail grep only `ValueError:|Traceback|ERROR|RuntimeError:`, stop means stop, findings only, no invented numbers. Ledger entries in `results/ledger.md` per milestone (`grok-4.6 · xai-oauth`). Box `ssh <user>@<station>`. Project `$HOME/hermes/pin-hot-experts/`. Read `HANDOFF-v17-seqs24-grok-4.6.md` and `results/v16/README.md` for context, but **this file overrides the v17 brief where they differ.**

## What happened (Milo, 2026-09-18 ~09:45 CDT)

The first v17 worker did everything right and was stopped by Milo because of two errors **in Milo's brief**:
1. The KV gate (≥ 1.8M) was arbitrary; v17 booted at **1,786,288 tokens** (1.70× at 1M), 14K under it. That container is the right one to test. **Gate is withdrawn.**
2. The fallback `--cudagraph-capture-sizes 1 2 4 8 12 16 24` was written as *sequence* counts; the flag takes **token** counts. With DSpark k=1 at ≥ 5 seqs a step is 2×seqs tokens, so everything above 12 seqs ran **uncaptured** — v17a C16 prose decode collapsed to 9.6 mean / 7.5 p10 tok/s (v15 ≈ 23 / 16). Its warm-turn numbers (C24 p50 0.67 s) are real but paid for with decode. That container is stopped-and-kept as `dsv41-vllm-v17b-seqs24-cg-EXP-TOKENSIZES-UNCAPTURED-FAIL`; its partial v17a receipts on the box (`$BOX_HOME/iwyzer/iw-v17a*.jsonl`, `run-v17a.log`) are v17b data — copy them to `results/v17/v17b-uncaptured/` and label them so, do not mix them into the v17 table.

**Real finding from v17b, keep it:** trimming capture sizes cut graph memory 2.33+1.78 GiB → 0.61+0.46 GiB and KV 1.79M → **2.96M** tokens (more than v15's 2.26M). A *correct* token list is a future lever; not this run.

## Your job: run the windows on v17 as booted

Container **`dsv41-vllm-v17-seqs24-EXP`** (seqs 24, lpt 6144, ksched `[[1,4,5],[5,24,1]]`, default graphs, KV 1,786,288) — Milo `docker start`ed it at ~09:45 CDT; it hash-hits and binds in ~6 min. Confirm `/v1/models` on `:30006`, the hook lines (`hot=295 cold=89`, `HBM_expert≈206.61`, `pinned_expert≈62.33`), and `GPU KV cache size: 1,786,288 tokens` in `$BOX_HOME/dsv41/dsv41-vllm-v17-seqs24-EXP.log`. **Do not launch anything new.** If it is not up or not binding, report and stop.

### 0. Box state
`docker ps` must show only `dsv41-vllm-v17-seqs24-EXP`. No `iwbench` processes (`pgrep -f "python3 [i]wbench"` — note the bracket; a plain `pkill -f iwbench` kills your own ssh shell). GLM stopped-and-kept; `:30003` down.

### 1. Idle probe on v17
`cd $BOX_HOME/dsv41 && BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none SIZES="8000 32000 128000" N=2 THINKING=0 python3 cold_prefill_probe.py` → `results/v17/cold_prefill_probe.out`. Reference (v15 idle): 16.8K / 22.3K / 22.0K tok/s, 4.72 s at 103.6K.

### 2. Same-window v17a → v15ctl2 → v17b
Per window, in order, from `$BOX_HOME/iwyzer/` (`BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none CHARS_PER_TOKEN=3.894 THINKING=0`):
`mix 8 120` (discard) · `mix 16 420` · `mix 24 420` · `coldload 5` · `tools 16` · then `$BOX_HOME/dsv41/knee.sh` ×2. Tags **`v17A`, `v15ctl2`, `v17B`** (capital A/B so they cannot collide with the v17b-uncaptured files); `OUT=iw-<tag>.jsonl`, logs `run-<tag>.log`. v15ctl2 = stop-and-keep v17, `docker start dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` (hash-hit). Stop-and-keep between windows; nothing else on `:30006` during a window. Copy all jsonl/logs/knee json to `results/v17/`.

### 3. Leave-up
All bars pass → v17 UP, v15 stopped-and-kept, **no renames** (Milo promotes). Otherwise `docker start` v15, rename v17 `-EXP-<reason>-FAIL`, never rm.

## Win bars (v17 two-window means vs v15ctl2)
- C24 warm agent turn TTFT p50: **≤ 5 s** (v15 was 27.1 s) — the point of the run
- C24 prose decode p10: report; no bar
- C16 warm agent TTFT p95: ≤ v15ctl2 + 0.5 s
- C16 chat TTFT p95: ≤ v15ctl2 × 1.1
- C16 prose decode p10: **≥ −5%** vs v15ctl2
- Cold 120K under 16 streams p95: ≤ v15ctl2 × 1.1
- Idle cold prefill 128K: ≥ −5% vs 22.0K tok/s
- knee C1 ≥ −1.5%; C8/C16 ≥ −5%
- Tools: ≥ v15ctl2's count (report both)
- Host Δ ≤ 2 GiB; no `ERROR|Traceback` in serving log

## Outputs
`results/v17/{README.md, iw-v17A*.jsonl, iw-v15ctl2*.jsonl, iw-v17B*.jsonl, run-*.log, knee-*.json, cold_prefill_probe.out, boot-log-excerpt.txt, v17b-uncaptured/}`, ledger entries. README must state both Milo errors above and the v17b graph-memory finding.
Verdict line: `V17 VERDICT: C24 warm p50 … s (v15 …) · C16 warm p95 … (v15 …) · chat p95 … · cold120K p95 … · prose p10 C16 Δ…% · idle 128K … tok/s · knee C1 Δ…% · tools …/64 vs …/64 · KV 1.786M · left up: v17|v15`. ≤ 15-line final summary, then stop.
