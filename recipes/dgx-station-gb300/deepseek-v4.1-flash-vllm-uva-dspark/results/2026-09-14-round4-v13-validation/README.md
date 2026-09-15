# Round 4 — v13 validation on real agent traffic; schedule-shape sweep — 2026-09-14 19:27–21:26 CDT

Operator plan: Milo, `round4/campaign_round4.sh` (here). All on the `:30006` lane; `:30003` untouched. Containers kept.
v13 = `dsv41-vllm-v13-1M-ksched-BOUND-REF`, v12 = `…-RETIRED-REF`. Same image, same offload, same KV.

## What this round asked

v13 ([`../2026-09-14-round3-ksched-depth/`](../2026-09-14-round3-ksched-depth/README.md)) was promoted on the prose knee. Three
questions were open: (A) is the −7% tool_json in the C1 fixture real; (B) does the schedule hold up on **real Hermes
agent transcripts**, single and concurrent; (C) what does the knee look like at C3 and C6, the concurrencies an agent
lane actually sits at; (D) are the schedule breakpoints right.

## A — tool_json −7%: fixture ordering, not v13 (closed)

`agent_fixture.sh` runs its five classes sequentially; tool_json is first and is the only class that passes a tool
schema. Run with tool_json **first**, **alone**, and in the **default** order, on both containers:

| tool_json tok/s (accept) | first | alone | default |
|---|---|---|---|
| v13 | 95.7 (0.88) | 183.9 (0.90) | 183.8 (0.90) |
| v12 | 95.3 (0.88) | 183.6 (0.90) | 183.7 (0.90) |

Identical on v12 and v13 in every position. The 145–159 range seen in Round 3 was the same class landing on a cold
tool-schema path after the warm-up call. Every other class is within 0.3 tok/s across the two containers.
**Closed: no v13 regression.** Lesson for the instrument: warm each prompt class before timing it.

## B — real agent transcripts (`replay_c.py`, 24 real Hermes turns, 12 with tool calls)

Concurrent replay: N workers pull turns from a shared queue; agg = Σ completion tokens / wall; acceptance from
`/metrics` deltas over the whole run.

| | agg tok/s | accept | accepted/step | tool-turn per-req | text-turn per-req | tool calls emitted |
|---|---|---|---|---|---|---|
| **v12 n=1** | 111.3 | 0.52 | 2.61 | 112.4 | 110.1 | 3/12 |
| **v13 n=1** | 116.5 | 0.55 | 2.76 | 104.1 | 126.2 | 4/12 |
| **v12 n=4** (k=5) | **235.9** | 0.54 | 2.70 | 59.8 | 59.7 | 8/24 |
| **v13 n=4** (k=1) | **207.0** | 0.85 | 0.87 | 52.8 | 52.7 | 8/24 |

n=1 is a wash (+5%, one pass, inside this instrument's ±17 tok/s). **n=4 is a real loss: −12%.** The mechanism is in the
accepted/step column: at k=1 the drafter can only ever hand back one token per step, and on agent text it is right
85% of the time, so the step yields 0.87 accepted tokens. At k=5 with 54% acceptance the same traffic yields 2.70 per
step. On *prose* (the knee) k=5's window costs 3.7× the unique experts for ~0.3 acceptance and k=1 wins; on *agent
text* the acceptance is high enough that the window pays for itself even at C4. **The prose knee and real agent
traffic disagree at C3–C4, and v13 was chosen on the knee.**

## C — knee with C3 and C6 (`knee6.sh`, same window)

| | C1 | C2 | C3 | C4 | C6 | C8 | C12 | C16 |
|---|---|---|---|---|---|---|---|---|
| v12 | 91.1 | 128.1 | 155.2 | 178.9 | 219.1 | 261.2 | 278.9 | 321.1 |
| v13 | 91.0 | 128.0 | **192.6** | **224.5** | **272.9** | **320.1** | **378.4** | **434.9** |
| Δ | −0.1% | −0.1% | +24% | +26% | +25% | +23% | +36% | +35% |

On prose the k=1 regime wins from the first step it applies (C3). Consistent with Round 3.

## D — breakpoint sweep (two boots, each vs a same-window v13 control on `knee6`)

| schedule | C1 | C2 | C3 | C4 | C6 | C8 | C12 | C16 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S1 `[[1,3,5],[4,16,1]]` | −0.4% | −0.4% | **−20.1%** | −2.4% | −1.1% | −0.7% | +8.1% | −0.6% | loss (C3 at k=5) |
| S2 `[[1,2,5],[3,4,3],[5,16,1]]` | +0.2% | −0.2% | **−8.7%** | −4.3% | +0.4% | −0.1% | −0.8% | −1.5% | loss (C3–C4 at k=3) |

On the prose knee, any k>1 at C3 loses. v13's breakpoints are right *for the knee*. Neither shape was run through the
agent replay (B); that is the missing measurement — see below.

**Autotune-hash finding.** S2 hit v13's hash `9ac7b387` (boot 453 s). S1 got a **new** hash `cbf5ea8f` (189 configs, 80-min
tune). Both have max k=5. So the FlashInfer autotune key includes the batch-size *breakpoints*, not just the maximum
speculative length — `[[1,2,5],…]` and `[[1,3,5],…]` retune, `[[1,2,5],[3,4,3],…]` does not. Round 3's "the schedule does
not change the hash" was true only for the one schedule tested. Budget 80 min for any new breakpoint set.

## Verdict

- **v13 stands as the recipe reference** for its stated workload (prose/mixed, throughput at load). A and C confirm it; D
  says its breakpoints are the right ones on that workload.
- **Caveat added to the recipe:** for a lane whose concurrent traffic is mostly agent text (Hermes tool/code turns) at
  3–4 streams, v13 measures −12% against v12 on real transcripts. That lane wants k=5 held to C4: `[[1,4,5],[5,16,1]]`,
  untested (new hash, one 80-min boot + replay at n=4 to decide). Single-stream agent lanes are unaffected.
- Not a rollback: the knee gain at C≥3 is +23–36% and the agent loss is one pass of one instrument. Confirm before acting.

## Files

- `knee-R4-*.json`, `agentfix-R4-*.json`, `replayc-R4-*.json` — raw per run
- `facts-R4-shapes.txt` — autotune hash / configs for S1 and S2
- `campaign-round4-2026-09-14.log`, `ledger-round4.md`
- `campaign_round4.sh`, `agent_fixture_o.sh` (CAT_ORDER), `knee6.sh` (adds C3/C6), `replay_c.py` (concurrent real-transcript replay)
