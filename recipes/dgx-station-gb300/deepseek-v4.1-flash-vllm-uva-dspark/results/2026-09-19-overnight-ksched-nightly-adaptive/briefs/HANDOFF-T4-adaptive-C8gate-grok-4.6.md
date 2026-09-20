# HANDOFF — T4: adaptive placement, unfrozen, gated at C8/C16 — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. All rules in `HANDOFF-grok-4.6.md` apply (never `:30003`, never `glm53-*`, never `docker rm`, never `docker pull`, drop caches before a launch, fast-fail grep `ValueError:|Traceback|ERROR|RuntimeError:`, stop means stop / `STOP-CAMPAIGN`, findings only, no invented numbers). Ledger `results/ledger.md` (`grok-4.6 · xai-oauth`). Box `ssh $BOX_USER@$BOX_HOST`. Mac project `$HOME/hermes/pin-hot-experts/`. Read `HANDOFF-e4b-grok-4.6.md` and `results/2026-09-17-e4-e4b-e5-placement-adaptation/{e4b,e5}/README.md` first. **Precondition: T3 finished, v18 back up on `:30006`.**

## Why

E4b measured the D1 counter tax at **C1 −1.68 / C8 −1.26 / C16 −0.94 %** and was failed on the C1 ≤ 1% bar, frozen (never swapped). E5 showed that a map profiled on mixed traffic wins held-out JP/DE **+64%** and math +10% but costs the agent knee −10/−16/−17% — the 89-slot budget is zero-sum across domains, and today the answer is "one map per traffic class, swap by reboot." James's question: *can we observe a little?* This run answers it where it is cheapest and where it can pay: **verdict at C8/C16, on v18 flags, unfrozen, with the off-profile held-out as the earn-back test.** Pass means "one lane, no per-domain reboot, ≤1.5% tax at C8" — an operational claim, not a speed claim. Fail is also a result: it closes adaptive placement on this hook for good.

One axis: `PIN_MODE=adaptive` (D1 counter, swaps live) on the v18 launch. Hook, rowmap seed, image, seqs 24, ksched, lpt, capture sizes, offload, util — all v18. Autotune: the hook adds no engine-config field; expect a hash **hit** on v18's `62426808…`; arm `seed_autotune.sh` anyway.

## Launch

`launch-t4-adaptive-v18.sh` (Mac `scripts/` → box `$BOX_HOME/pin-hot-experts/scripts/`). It is `launch-many-seat.sh` + the E4b adaptive env (`PIN_COUNTER=d1`, `PIN_ADAPTIVE_RING=0`, `PIN_SWAPS_PER_LAYER_PER_DRAIN=2`, `PIN_EWMA_HALFLIFE=500`, `PIN_DRAIN_N=64`), **`PIN_ADAPTIVE_FREEZE=0`**, drain parked on `$BOX_HOME/pin-hot-experts/t4/GO`. The GO file is NOT created by the launcher. Container `dsv41-vllm-T4-adaptive-d1-unfrozen-C8gate-EXP`.

Stop-and-keep v18. Drop caches. Launch. Wait for `/v1/models`. Verify hook lines (`hot=295 cold=89`, `HBM_expert=206.61GiB`), `ADAPT counter=d1 … layers=40`, `worker parked until GO file`, KV 2,502,950 ±0.5%, graphs 1.30/1.10 GiB, hash hit. **Only after bind and one `smoke_vllm.sh` pass:** `touch $BOX_HOME/pin-hot-experts/t4/GO`. Confirm `GO seen freeze=False` in the log.

## Windows: T4 → v18ctl → T4b (stop-and-keep between)

Per window, from `$BOX_HOME/dsv41/`, `BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva`:

1. **Warm the adaptive state on the lane's own traffic first** (T4 windows only): `replay_c.py` n=4 ×1, discarded. Record `n_swaps` from `t4/swaps.jsonl` after it (expect few — v18's map is already the agent profile).
2. `knee.sh` ×2 → `knee-<tag>-r{1,2}.json` — **verdict instrument for C8 and C16; C1 reported only**
3. `CAT_ORDER="prose structured code shell_ops tool_json" agent_fixture_o.sh` → `agentfix-<tag>.json`
4. **Held-out, the earn-back test:** `python3 $HOME/hermes/pin-hot-experts/scripts/e2c_heldout.py` (5 math + 5 JP/DE, streaming first→last, 400 tok) → `heldout-<tag>.json`. On T4 windows run it **twice back-to-back**: pass 1 is "cold" (map still agent-shaped), pass 2 is after the drain has seen JP/DE. Report both. Record `n_swaps` before/after, and cold-churn (distinct expert ids swapped) from `swaps.jsonl`.
5. Greedy parity: `python3 scripts/e2c_parity.py` on the 18-prompt set (`scripts/e5_parity_prompts.json`) after ≥ 200 total swaps have been applied (if fewer than 200 by the end of T4b, run it anyway and record the count) → `parity-<tag>.json`. Compare T4b vs v18ctl.
6. `curl -s :30006/metrics | grep spec_decode` → `metrics-<tag>.txt`; cgroup `memory.current` start/end → `hostwatch-<tag>-{start,end}.json`.
7. Copy `t4/swaps.jsonl` to `results/2026-09-19-overnight/T4/swaps-<tag>.jsonl` at the end of each T4 window.

Do not run anything else against `:30006` during a window.

## Win bars (T4/T4b two-window means vs v18ctl)

- **C8 ≥ −1.5%**, **C16 ≥ −1.0%** on `knee.sh` (the measured tax was −1.26/−0.94; the bar asks that live swaps do not add to it). C1 reported, not gated.
- **Held-out JP/DE pass 2 ≥ +20%** over v18ctl (E5 got +64% with a rebuild; adaptive should collect a substantial fraction without one). Math pass 2 ≥ −1.5%.
- Agent fixture: no class < −3% (tool_json fixture-order jitter is known; if only tool_json fails and by < 7%, note it, do not fail on it alone).
- Parity 18/18 content+tool-args (ids ignored) T4b vs v18ctl.
- Swaps: plateau observed (swaps per drain falls over the window); `n_swaps` reported; no `ERROR|Traceback`; host Δ ≤ 2 GiB.

All pass → leave **T4 UP**, v18 stopped-and-kept; Milo promotes. Any fail → `-EXP-<reason>-FAIL`, restore v18. A held-out win with a C8 miss is still a FAIL for promotion but is the key number for the write-up — report it prominently.

## Outputs

`results/2026-09-19-overnight/T4/{README.md, boot-log-excerpt.txt, knee-*.json, agentfix-*.json, heldout-*.json, parity-*.json, metrics-*.txt, hostwatch-*.json, swaps-*.jsonl}`, ledger entries. Verdict:
`T4 VERDICT: knee C8/C16 Δ…/…% (C1 Δ…%) · held-out JP/DE p1 … p2 … vs v18 … (Δ…%) · math p2 Δ…% · swaps n=… churn … plateau yes/no · parity …/18 · fixture min Δ…% · hash hit yes/no · left up: T4|v18`
≤ 15-line final summary, then stop.
