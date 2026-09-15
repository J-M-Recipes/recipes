# Round 6 — instrument v2, prefix-cache under real session shape, offload floor — 2026-09-15 04:08–05:49 CDT

Operator: Milo. All on `:30006`. v14 (`[[1,4,5],[5,16,1]]`) and v13 containers; one OFFGB=55 boot attempt. `:30003` untouched. Lane dark at end.

## What was asked

1. Make the real-transcript replay trustworthy: Rounds 4–5 used 24 turns and swung ±20% run to run.
2. Measure what agents actually wait on — prefix-cache hit rate and TTFT — not just decode tok/s.
3. Try buying HBM back from the expert offload.

## 1. Instrument v2: 300 real turns, bootstrap CI (`replay2.py`, `build_fixture_v2.py`)

`transcript_fixture_v2.json` — 300 assistant turns sampled from Hermes `state.db` (67,926 candidates), stratified 150 tool-call / 150 text, ≤2 turns per session (229 sessions), context capped ~6K tokens (oldest messages dropped, first user message kept), secrets scrubbed (59 redactions). Fixture is not in this bundle (private transcripts); the builder is.

4 workers, shared queue, streaming, `max_tokens=400`, thinking off, tools offered. One pass ≈ 7 min.

| | run a | run b | CI95 (bootstrap, per-request) | accept | acc/step |
|---|---|---|---|---|---|
| v14 | 171.1 | 174.1 | ±5 | 0.56 | 2.8 |
| v13 | 150.0 | 151.7 | ±3 | 0.87 | 0.87 |

Run-to-run spread ±1% (was ±20% on 24 turns). **v14 vs v13 on agent traffic: +14%**, unambiguous; Round 5's 24-turn version said +10% with overlapping bars. Confirms v14 as the agent-lane schedule.

**But:** prefix-cache hit rate on this fixture was **5%**, 2.09M prompt tokens for 68K generated. That is a property of the fixture — every turn from a different session, so almost no shared prefix — not of the lane. Every agent-replay number in Rounds 4–6a (this table included) measures decode under a prefill-dominated, cache-defeating load. They rank configs correctly (same load both sides) but understate the lane. Hence §2.

## 2. Session-ordered replay (`replay_sessions.py`, `build_fixture_sessions.py`) — the honest agent number

`transcript_fixture_sessions.json` — 20 sessions (of 754 eligible with ≥15 assistant turns), first 15 turns each in order, 300 turns, 277 with tool calls (that is what the first 15 turns of a real Hermes session look like). Each turn's context is the full prefix, capped at 60K chars by dropping the middle. Prefix grows from ~900 to ~13K prompt tokens by turn 14. 4 concurrent sessions, turns serial within a session.

| v14 | agg tok/s (wall) | prefix-cache hit | TTFT med / p90 | TTFT t0 → t7 → t14 | accept | acc/step |
|---|---|---|---|---|---|---|
| run a (cold cache, fresh boot) | **232.6** | **71.3%** | **0.45 / 0.77 s** | 0.34 → 0.41 → 0.56 s | 0.69 | **3.46** |
| run b (warm — same sessions again) | 275.2 | 94.3% | 0.32 / 0.47 s | 0.23 → 0.33 → 0.32 s | 0.69 | 3.46 |

Run a is the production-shaped number: a new session's turns hit cache 71% of the time (everything but the last exchange), TTFT p90 under 0.8 s, and TTFT grows only 0.34 → 0.56 s as context reaches 13K tokens. Run b is what a lane looks like when the same sessions continue after a warm-up — the ceiling. Acceptance is higher on in-session continuation (3.46/step) than on the shuffled fixture (2.8) — the drafter does better on text it has context for.

**The shuffled-fixture number for v14 was 171; the session-shaped number is 233 (cold) to 275 (warm).** The 24-turn Round 5 figure (207) sat between by accident.

Estimator note: the bootstrap CI in `replays-*.json` (`agg_ci95_boot`, per-request resample × N workers) assumes all workers busy for the whole wall; with session-serial workers the tail idles, so it lands ~3–7% above the wall-clock agg and is reported here as a spread, not a location. `agg_tok_s` (wall-clock) is the number.

## 3. OFFGB=55: does not bind

v14 flags with `--cpu-offload-gb 55` (5 GiB more experts in HBM). Two allocator OOMs during bind; `Available KV cache memory: 0.65 GiB` (v14 at 60: 4.87 GiB). At util 0.97 the weights fill HBM; **60 GiB is the offload floor** on this box with this image. KV at 1M is only ~5 GiB, so shrinking `CTX` does not free meaningful HBM either — the expected "256K lane with 30 GiB more experts resident" does not exist. `facts-R6-off55.txt`.

## Files

- `replay2-R6-{v14,v13}-{a,b}-n4.json` — shuffled-fixture runs, per-request records included.
- `replays-R6-sess-v14-{a,b}-n4.json` — session-ordered runs, per-request records, TTFT by turn index.
- `facts-R6-off55.txt` — the failed bind.
- `campaign-round6-2026-09-15.log`, `campaign-round6b-2026-09-15.log`, `campaign_round6.sh`, `run_sessions.sh` — receipts.
- `replay2.py`, `replay_sessions.py`, `build_fixture_v2.py`, `build_fixture_sessions.py` — the instruments. Fixtures are built from the operator's own Hermes `state.db`; build your own.

## What this changes

- Recipe metric "real-agent replay" now cites the session-ordered cold number (233) with the shuffled number kept as the config-ranking instrument.
- New `limits` entries: offload floor 60 GiB; shuffled replay understates the lane.
- No config change. v14 stays.
