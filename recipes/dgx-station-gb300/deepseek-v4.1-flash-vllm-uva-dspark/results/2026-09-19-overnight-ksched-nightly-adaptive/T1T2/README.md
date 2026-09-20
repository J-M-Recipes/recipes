# T1 / T2 — k-schedule under the v15 map — FAIL (both)

Runner: Miloh's `overnight-runner-2026-09-18.sh` (nohup). Windows 21:53–04:57 CDT 2026-09-19. Launch = v18 (`launch-many-seat.sh`) with only `KSCHED` changed. Hash hit `62426808…` both (Loaded 189 / 0 new, ~8-min binds). Hook lines verified 40×295/89, `HBM_expert=206.61 pinned_expert=62.33`. KV: T1 2,406,686 (−3.8% vs v18 — k=3 mid-band graphs cost HBM), T2 similar.

Sequence: v18ctl → T1 → T2 → v18ctl2 → T1b → T2b (two windows each, two controls). Receipts: `receipts/window-*.log`, `knee-*.json`, `knee6-*.json`, `agentfix-*.json`, `replayc-*.json`, `metrics-*.txt`.

## Two-window means vs v18ctl (v18ctl + v18ctl2)

| | C1 | C8 | C16 | replay n=4 | prose | tool_json | verdict |
|---|---|---|---|---|---|---|---|
| v18ctl `[[1,4,5],[5,24,1]]` | 171.9 | 657.0 | 847.2 | 395 | 202 | 241 | — |
| T1 `[[1,4,5],[5,12,3],[13,24,1]]` | 170.7 (−0.7%) | **616.0 (−6.2%)** | 847.7 (+0.1%) | 380 (−3.8%) | 196 (−3.0%) | 225 (−6.5%) | **FAIL** |
| T2 `[[1,8,5],[9,24,3]]` | 171.3 (−0.3%) | **572.3 (−12.9%)** | **795.9 (−6.1%)** | 385 (−2.7%) | 197 (−2.5%) | 227 (−5.6%) | **FAIL** |

Bars: C8 ≥ +5% and C16 ≥ +5%, or replay ≥ +8% with C8/C16 ≥ −2%; no fixture class < −3%. Neither is close on any bar.

## Why (from the receipts, not the hypothesis)
- Accepted-per-step is 2.6–2.7 on every run including v18 (`replay … acc/step`). The replay's four workers never leave the k=5 band (≤4 running seqs), so a k=3 band from 5 or 9 seqs is never exercised where it was supposed to help; on the prose knee at C8 it is pure verify-compute tax (k=3 streams 1.5× the tokens of k=1 into a step that already fills).
- Fixture acceptance is identical to the control on every class (k=5 band unchanged) — the tool_json −6% is the C1 fixture running on a slightly slower box state, not the schedule; the C1 knee is −0.3/−0.7%.
- Miloh's overnight read (close the k-schedule lever on this lane) stands. The external spark-bench k=3 win was on chat at C1–C4, where this lane already runs k=5.

VERDICT T1: FAIL — C8 −6.2% · C16 +0.1% · replay −3.8% · hash hit · stopped-and-kept `dsv41-vllm-T1-ksched-5-3-1-EXP`.
VERDICT T2: FAIL — C8 −12.9% · C16 −6.1% · replay −2.7% · hash hit · stopped-and-kept `dsv41-vllm-T2-ksched-5-3-EXP`.
