# HANDOFF — v18-cgsizes: recover KV on the v17 many-seat profile with token-correct capture sizes — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. Read fully before acting. All rules in `HANDOFF-grok-4.6.md` apply: never `:30003`, never start GLM (`glm53-*`), never `docker rm`, never `docker pull`, drop caches before a fresh launch, fast-fail grep only `ValueError:|Traceback|ERROR|RuntimeError:`, stop means stop, findings only, no invented numbers. Ledger entries in `results/ledger.md` per milestone (`grok-4.6 · xai-oauth`). Box `ssh <user>@<station>`. Project `$HOME/hermes/pin-hot-experts/`. Read `results/v17/README.md` first (it documents the v17b token-size mistake this run corrects), then `results/v16/README.md`.

## State (Milo, 2026-09-18 ~11:50 CDT)

- **`:30006` reference is now v17:** `dsv41-vllm-v17-seqs24-BOUND-REF` (seqs 24, lpt 6144, ksched `[[1,4,5],[5,24,1]]`, default graphs, KV **1,786,288**). James promoted it for the many-seat latency profile. It is your **control** this run.
- v15 kept as `dsv41-vllm-v15-1M-pin-static-v1-RETIRED-REF` (stopped). v17b kept as `...-TOKENSIZES-UNCAPTURED-FAIL` (stopped).
- v17's costs: default 24-seq graphs took 2.33 + 1.78 GiB → KV fell 2.26M → 1.79M (1.7× at 1M); knee C8 −7% vs v15 (unexplained); C16 prose p10 −12% vs v15 on a single control window (unconfirmed).
- v17b (wrong list `1 2 4 8 12 16 24`) proved trimmed capture sizes free ~3 GiB (graphs 0.61 + 0.46 GiB, KV 2.96M) but left >12 seqs uncaptured because the flag takes **token** counts and DSpark k=1 makes a step 2×seqs tokens.

## One axis: `--cudagraph-capture-sizes` (token counts)

Candidate `v18` = v17 launch + `CGSIZES="1 2 4 6 8 12 16 18 24 32 40 48 64 96 128"`. Coverage: k=5 at 1–4 seqs = 6/12/18/24 tokens (exact hits); k=1 at 5–24 seqs = 10–48 tokens (pad ≤ 8); 64/96/128 keep small mixed prefill+decode steps captured. Everything else identical to v17.

Launch: `NAME=dsv41-vllm-v18-cgsizes-EXP CGSIZES="1 2 4 6 8 12 16 18 24 32 40 48 64 96 128" $BOX_HOME/pin-hot-experts/scripts/launch-v16-sched.sh` (defaults are already v17: SEQS=24 LPT=6144). Verify in `docker inspect … .Args` that `--cudagraph-capture-sizes` carries exactly those 15 values and `--max-num-seqs 24`.

Expect: autotune **hit** likely (v17b hit with the same seqs; capture sizes are not kernel shapes) — if it tunes, record minutes, it is not a failure. Hook lines required (`hot=295 cold=89`, `HBM_expert≈206.61`, `pinned_expert≈62.33`). Record `Graph capturing finished … took N GiB` (both lines) and `GPU KV cache size`. **No KV gate** — record and proceed. If it fails to boot, capture the error, and try once more with `CGSIZES="1 2 4 6 8 12 16 18 24 32 40 48"` as `dsv41-vllm-v18b-cgsizes12-EXP`; if that also fails, restore v17 and report.

## Procedure

0. Box state: only `dsv41-vllm-v17-seqs24-BOUND-REF` running. No `iwbench` (`pgrep -f "python3 [i]wbench"` — bracket pattern; a plain `pkill -f iwbench` kills your own ssh shell). GLM stopped-and-kept; `:30003` down. Record `nvidia-smi` GPU1 + `MemAvailable`.
1. Stop-and-keep v17. Drop caches. Launch v18. Wait for `/v1/models`. Record everything in the Expect paragraph.
2. Idle probe on v18: `cd $BOX_HOME/dsv41 && BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none SIZES="8000 32000 128000" N=2 THINKING=0 python3 cold_prefill_probe.py`. v17 idle: 12.1K / 17.5K / 23.3K tok/s (4.46 s at 103.6K).
3. Same-window **v18A → v17ctl → v18B → v15ctl3**. Per window, in order, from `$BOX_HOME/iwyzer/` (`BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none CHARS_PER_TOKEN=3.894 THINKING=0`):
   `mix 8 120` (discard) · `mix 16 420` · `mix 24 420` · `coldload 5` · `tools 16` · then `$BOX_HOME/dsv41/knee.sh` ×2. Tags exactly `v18A`, `v17ctl`, `v18B`, `v15ctl3`; `OUT=iw-<tag>.jsonl`, logs `run-<tag>.log`. Controls = stop-and-keep the running container, `docker start` the named REF (hash-hit, ~6 min). **The v15ctl3 window is for one question only:** a second v15 reading of C16 prose p10 and knee C8, to settle whether v17's −12% / −7% vs v15 were real. Run it last; if the box or time is a problem, skip it and say so.
4. Leave-up: all bars pass → v18 UP, v17 stopped-and-kept, **no renames** (Milo promotes). Otherwise `docker start` v17 `-BOUND-REF`, rename v18 `-EXP-<reason>-FAIL`, never rm.

## Win bars (v18 two-window means vs v17ctl)
- **KV ≥ 2.4M tokens** (the point of the run; v17 1.79M, v17b 2.96M)
- knee C1 / C8 / C16: each ≥ −1.5% (capture list must not cost decode — this is the check v17b failed)
- C16 prose decode p10: ≥ −5%
- C24 warm agent p50: ≤ 5 s; C16 warm p95: ≤ v17ctl + 0.5 s; chat p95 and cold120K p95: ≤ v17ctl × 1.1
- Idle 128K: ≥ −5% vs 23.3K tok/s
- Tools ≥ v17ctl's count
- Host Δ ≤ 2 GiB; no `ERROR|Traceback`
- Report separately (no bar): v15ctl3 C16 prose p10 and knee C8 next to v17ctl's and `results/v17/` numbers.

## Outputs
`results/v18/{README.md, iw-*.jsonl, run-*.log, knee-*.json, cold_prefill_probe.out, boot-log-excerpt.txt}`, ledger entries.
Verdict line: `V18 VERDICT: KV … M tok (v17 1.786M) · graphs … GiB · knee C1/C8/C16 Δ…/…/…% · prose p10 C16 Δ…% · C24 warm p50 … s · cold120K p95 … · idle 128K … tok/s · tools …/64 vs …/64 · hash hit yes/no · v15ctl3 prose p10 … / knee C8 … · left up: v18|v17`. ≤ 15-line final summary, then stop.
