# HANDOFF — T1+T2: k-schedule re-sweep under the v15 map — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. Read fully before acting. All rules in `HANDOFF-grok-4.6.md` apply: never `:30003`, never start GLM (`glm53-*`), never `docker rm`, never `docker pull`, drop caches before a fresh launch (`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`), fast-fail grep only `ValueError:|Traceback|ERROR|RuntimeError:`, stop means stop (a file `$BOX_HOME/dsv41/STOP-CAMPAIGN` also means stop between steps), findings only, no invented numbers. Ledger entries in `results/ledger.md` per milestone (`grok-4.6 · xai-oauth`). Box: `ssh $BOX_USER@$BOX_HOST`. Project on the Mac: `$HOME/hermes/pin-hot-experts/` (run from here; write results here). Plan of record: `OVERNIGHT-PLAN-2026-09-18.md`.

**A `docker pull` of the nightly image may be running in the background on the box** (`$BOX_HOME/dsv41/pull-nightly-dee37d89.log`, started by Milo, approved by James). Leave it alone. It is disk/network only.

## Why

v18's k-schedule `[[1,4,5],[5,24,1]]` forces k=1 from 5 streams up. That rule was set 2026-09-12–14 **under positional offload**, on the reasoning that a k=5 verify window streams 3.7× the unique experts, a quarter of them over the 340 GB/s link. Under the v15 usage map that quarter is ~1.4% (`results/2026-09-12-v12-1M-k5-off60-util97/routing/README.md` last column), and computed on the E2a routing profile the k=5:k=1 byte ratio at C16 collapses from 2.08× to 1.80×. The byte argument for k=1 at high concurrency is gone; the accept-length argument for k≥3 on agent text (85% accept) is unchanged. Externally, neko-legends/spark-bench measured k=3 beating k=5 on chat (C1 30.2→34.4, agg@4 58.6→67.8). This run tests two schedules. **One axis: the K values.** Hook, rowmap, image, seqs, lpt, capture sizes, offload, util — all v18.

Hash: the FlashInfer autotune key does not include `num_speculative_tokens_per_batch_size` (`vllm/config/speculative.py:600-646`, verified on the 0909 image). Expect a hash **hit** (`Loaded N configs`, ~6 min bind). The campaign once saw `[[1,3,5],[4,16,1]]` retune anyway — so arm `seed_autotune.sh` before each launch regardless and treat a miss as expected-not-failure (budget 80 min, record it).

## Launches

From the box, `cd $BOX_HOME/pin-hot-experts/scripts` (copy `launch-many-seat.sh` there from `results/2026-09-18-many-seat-v16-v17-v18/` if absent; it is the v18 launcher; `BOX_HOME=$BOX_HOME`).

| tag | container name | KSCHED |
|---|---|---|
| T1 | `dsv41-vllm-T1-ksched-5-3-1-EXP` | `[[1,4,5],[5,12,3],[13,24,1]]` |
| T2 | `dsv41-vllm-T2-ksched-5-3-EXP` | `[[1,8,5],[9,24,3]]` |

```
NAME=dsv41-vllm-T1-ksched-5-3-1-EXP KSCHED='[[1,4,5],[5,12,3],[13,24,1]]' bash launch-many-seat.sh
```
(and the T2 line with its NAME/KSCHED). Everything else defaults (= v18).

Verify per boot, paste into the README: `PIN_HOT rowmap layer … hot=295 cold=89`, `HBM_expert=206.61GiB pinned_expert=62.33GiB`, `GPU KV cache size: 2,502,950 tokens` (±0.5%), `Graph capturing finished … 1.30 GiB` then `1.10 GiB`, autotune hash id + `Loaded N configs` or tune minutes, bind time. A boot missing the hook lines is positional offload — stop and report.

## Procedure

0. Box state: `docker ps` shows only `dsv41-vllm-v18-cgsizes-BOUND-REF`. Record `nvidia-smi` GPU1 used/total and `MemAvailable`.
1. Stop-and-keep v18 (`docker stop`, no rename). Drop caches. Launch **T1**. Wait for `/v1/models` on `:30006`.
2. Windows, in this order, stop-and-keep between each: **T1 → v18ctl → T2 → v18ctl2 → T1b → T2b**. v18ctl = `docker start` the `-BOUND-REF` (hash-hit, ~5 min). T1b/T2b = `docker start` the kept T1/T2. Six windows; the two controls are shared by both candidates.
3. Per window, from `$BOX_HOME/dsv41/` with `BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva`:
   - `knee.sh` ×2 → `knee-<tag>-r{1,2}.json` (verdict instrument)
   - `knee6.sh` ×1 → `knee6-<tag>.json` (C3/C4/C6/C12 — the band the schedule changes)
   - `CAT_ORDER="prose structured code shell_ops tool_json" agent_fixture_o.sh` → `agentfix-<tag>.json`
   - `replay_c.py` n=4 ×2 → `replayc-<tag>-r{1,2}-n4.json` (report accept rate and accepted/step from its output)
   - after replay: `curl -s :30006/metrics | grep spec_decode` → `metrics-<tag>.txt`
   Do not run anything else against `:30006` during a window.
4. Copy all json/txt + boot-log excerpts to `results/2026-09-19-overnight/T1/` and `T2/`; write one `README.md` per candidate with the tables below.
5. Leave up: **v18** (`docker start` the BOUND-REF) unless a candidate passes its promotion rule — then leave that candidate UP and v18 stopped-and-kept. Do not rename REFs; rename failed candidates `-EXP-<reason>-FAIL`. Milo promotes.

## Win bars (two-window means per candidate vs the two v18ctl windows)

- C1 knee: ≥ −1% (sanity — the k=5 band is identical)
- **Promote if:** (C8 ≥ +5% **and** C16 ≥ +5% on `knee.sh`) **or** (replay n=4 mean ≥ +8% **and** C8, C16 ≥ −2%)
- No `agent_fixture_o` class < −3%
- replay tool turns parse (tool_calls present where v18ctl has them)
- No `ERROR|Traceback` in the serving log during windows; host cgroup Δ ≤ 2 GiB per window
- If both pass, promote the one with the higher replay mean; tie → higher C16.

State the expected asymmetry in the README: prose knee at C8/C16 may lose (k=3 verifies 2× the tokens of k=1 into a batch that already fills the step) while replay wins (agent text accepts ~85%). Report both; the promotion rule encodes the lane's real traffic.

## Outputs

`results/2026-09-19-overnight/{T1,T2}/{README.md, knee-*.json, knee6-*.json, agentfix-*.json, replayc-*.json, metrics-*.txt, boot-log-excerpt.txt}`, ledger entries. Verdict lines:
`T1 VERDICT: knee C1/C8/C16 Δ…/…/…% · knee6 C3/C4/C6/C12 Δ… · replay n=4 mean … vs … (Δ…%) accept … acc/step … · fixture min class Δ…% · hash hit yes/no (… min) · KV … · left up: T1|v18`
same for T2. ≤ 15-line final summary, then stop.
