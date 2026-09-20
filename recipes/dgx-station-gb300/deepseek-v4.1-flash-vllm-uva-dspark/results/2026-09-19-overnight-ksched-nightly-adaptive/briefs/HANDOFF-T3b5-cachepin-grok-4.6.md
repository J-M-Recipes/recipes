# HANDOFF — T3b5: cache-pin discriminator (nightly + v15 hook) — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. All rules in `HANDOFF-grok-4.6.md` apply (never `:30003`, never `dsfv-*`, never `glm53-*`, never `docker rm`, never `docker pull`, stop means stop / `STOP-CAMPAIGN`, findings only, no invented numbers). Ledger `results/ledger.md`, sign entries `grok-4.6 · xai-oauth`. Box `ssh $BOX_USER@$BOX_HOST`. Mac project `$HOME/hermes/pin-hot-experts/`. **Read first:** `results/2026-09-19-overnight/T3b/README.md` (the state this test resolves) and `results/2026-09-19-overnight/README.md` (the one table).

## Why

T3b (today, daytime): the nightly `dee37d89` + v15 hook lane measures **C1 184–196 tok/s vs v18 172 (+7% / +14%)** on boots where FlashInfer autotunes live in-process — and **exactly 172 with a C4 −15% hole** on any boot that *loads* a saved autotune cache. Two live tunes chose different configs (27/42 MoE, 58/147 GEMM entries) and the saved caches reproduced a slow set. Untested: whether the *fast* set (`ed692e15….from-T3b4`, the 196 boot's own save) reproduces 196 when loaded.

This is one boot and it forks the recipe:
- **~196 (≥ +10% vs control):** config selection is the whole story → recipe = pin a known-good autotune file. Promotion candidate.
- **~172 (≈ control):** the cache-*load path itself* is the tax, independent of which configs are loaded → the recipe is "serve from a live tune" and the finding goes upstream to FlashInfer with both JSONs.
- **Anything between:** report it; do not interpret.

## What runs (already written — do not rewrite the runner)

`scripts/t3b5-cachepin-runner-2026-09-19.sh` (Mac → box `$BOX_HOME/pin-hot-experts/scripts/`). It:
1. `safety` + `memcheck`; moves the slow set aside, **copies** `…from-T3b4` into the canonical hash dir (sha256 of both recorded to `overnight-T3/cachepin-sha-T3b5.txt`); the original `.from-T3b4` is never modified.
2. Stop-and-keep v18, drop caches, `HOOK=1 OFFGB=54 NAME=dsv41-vllm-T3b5-…-CACHEPIN-EXP launch-t3-nightly.sh`. Expect **hash hit**: `Loaded 189 configs` and an ~8–10 min bind, *not* a 15–17 min tune. If the log says `Saved` / `Autotuning process ends` with a long tune, the pin did not take — that is a finding, record it.
3. Window T3b5 (`run_window_T1T2.sh`: knee ×2, knee6, agent_fixture_o, replay n=4 ×2), parity 18, `nvidia-smi` power/clock line before+after each window (new instrument — first time on this lane).
4. Control: `docker start` v18 → window v18ctl6 → parity.
5. T3b5b: second fresh `docker run` with the same fast cache → window → parity.
6. Restore: post-run cache copy kept as `.after-T3b5`, slow set back at the canonical path, v18 restarted on `:30006`. Nothing deleted.

Everything appends to `$BOX_HOME/dsv41/overnight-campaign-2026-09-18.log`; window receipts land in `$BOX_HOME/dsv41/overnight-T3/`, parity in `$BOX_HOME/pin-hot-experts/e2c/t3b-parity/`.

## Your job

1. **Stage:** `scp` the runner to the box path above; `chmod +x`. Confirm preconditions on the box: only `dsv41-vllm-v18-cgsizes-BOUND-REF` running; `ls …/103a/` shows the three dirs (`bba7410c…`, `ed692e15…`, `ed692e15….from-T3b4`); no `STOP-CAMPAIGN` file; `MemAvailable` > 100 GiB.
2. **Launch detached:** `nohup bash $BOX_HOME/pin-hot-experts/scripts/t3b5-cachepin-runner-2026-09-19.sh > $BOX_HOME/dsv41/t3b5-nohup.out 2>&1 &` then `disown`. Never run it in the foreground of an ssh session.
3. **Poll**, don't block: every 2–3 min `tail -5` the campaign log; `ps -eo args | grep '^bash $BOX_HOME/pin-hot-experts/scripts/t3b5'` (not `pgrep -f`). Budget ~1.5 h total (three ~8-min binds + three ~12-min windows + parity). Nothing else against `:30006` while it runs.
4. **Read back** — every number from the JSON/log, none from memory:
   - `overnight-T3/boot-T3b5.txt` / `boot-T3b5b.txt`: `Loaded N configs` vs `Saved`; bind seconds; `HBM_expert=206.61GiB pinned_expert=62.33GiB`; `GPU KV cache size`.
   - `overnight-T3/knee-{T3b5,v18ctl6,T3b5b}-r{1,2}.json`, `knee6-*.json`, `agentfix-*.json`, `replayc-*.json`, `metrics-*.txt`. C1 = knee r1/r2/knee6; C8, C16 = r1/r2 (pair r1/r1, r2/r2 only — r1 is ~25% under r2 on every window).
   - parity: `e2c/t3b-parity/parity-*.json`. Compare **T3b5 vs T3b5b** (same cache, two boots: expect 16/18 = only tool prompts jitter) and **T3b5 vs T3b4** if `parity-T3b4.json` exists (same configs loaded vs the tune that made them: text prompts should be 16/16 identical). Do **not** apply an 18/18 bar vs v18 — cross-image parity is meaningless (documented).
   - power lines from the campaign log (`power-pre`/`power-post`/`power-ref`): report draw/cap/SM clock; flag any throttle reason ≠ `0x0`.
5. **Copy receipts** to Mac `results/2026-09-19-overnight/T3b5/receipts/` (window logs, json, metrics, boot/smoke txt, parity, `cachepin-sha-T3b5.txt`, the three `autotune_configs.json` with their sha256s).
6. **Write** `results/2026-09-19-overnight/T3b5/README.md`: one table (rows T3b5 / v18ctl6 / T3b5b, plus T3b4 and v18ctl5 copied from the T3b README for reference), boot facts, parity table, power table, then a **VERDICT T3b5:** line in the ledger style. Append ledger entries per milestone (`grok-4.6 · xai-oauth`).

## Bars (report, don't promote — Milo promotes)

- **Hash hit** (`Loaded 189`, no live tune) — mandatory for the result to mean anything.
- **CACHE-PIN CONFIRMED:** C1 two-window mean ≥ +10% vs v18ctl6 **and** C4 ≥ v18ctl6 −3% (the loaded-cache hole was C4 −15%) **and** T3b5 vs T3b5b within 2% at C1.
- **LOAD-PATH TAX CONFIRMED:** C1 within ±2% of v18ctl6 with the C4/C12 hole present.
- Either outcome is a clean finding. Residency lines must be identical to v18 (206.61/62.33); KV expected ~4.53M on the nightly.

## Stop / failure

Boot fail → the runner renames `-BOOT-FAIL`, restores caches and v18, exits; record the exact error from `docker logs`, do not retry. If v18 does not come back (`/v1/models` on `:30006` has no `dsv41-flash-uva` after 15 min): `docker ps -a | grep dsv41`, then `docker start dsv41-vllm-v18-cgsizes-BOUND-REF` once, report, stop. `touch $BOX_HOME/dsv41/STOP-CAMPAIGN` stops it between steps. Do not touch `.from-T3b4`, `bba7410c…`, or any container not named in this brief.

End with the VERDICT line and a ≤15-line summary: which fork, the C1/C4/C8/C16/replay numbers, parity counts, power line, what is up on `:30006`.
