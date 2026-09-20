# HANDOFF — T3b6: KV-dtype discriminator (nightly + hook + pinned cache + `fp8_ds_mla`) — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. All rules in `HANDOFF-grok-4.6.md` apply (never `:30003`, never `dsfv-*`, never `glm53-*`, never `docker rm`, never `docker pull`, stop means stop / `STOP-CAMPAIGN`, findings only, no invented numbers). Ledger `results/ledger.md`, sign `grok-4.6 · xai-oauth`. Box `ssh $BOX_USER@$BOX_HOST`. Mac project `$HOME/hermes/pin-hot-experts/`. **Read first:** `results/2026-09-19-overnight/T3b5/README.md` (the run this one is one flag away from) and `HANDOFF-T3b5-cachepin-grok-4.6.md` (procedure — identical except where stated here).

## Why

T3b5 showed the nightly `dee37d89` + v15 hook + pinned autotune set is +9.6% C1 over v18 but **loses on the agent fixture**: prose −10.5%, tool_json −10%, structured −8% — because the nightly accepts fewer DSpark drafts on every boot (prose 0.288 vs 0.349, tool_json 0.81 vs 0.88). Milo found the likely cause in the boot logs: the nightly **silently switched the KV cache format from `fp8_ds_mla` (0909) to `nvfp4_ds_mla`** (`attention.py:156 Using DeepSeek's nvfp4_ds_mla KV cache format`, from vLLM #56935, SM100-only). That is also why KV reports 4.53M tokens instead of 2.50M — half the bytes per token. A 4-bit NoPE latent changes the target logits the drafter is verified against. `fp8_ds_mla` is still accepted by the nightly's backend.

One flag, three outcomes:
- **acceptance returns to ~0.35/0.88 and C1 stays ≥ ~183** → the first nightly config that wins on every class; promotion candidate.
- **acceptance returns but C1 drops to ~172** → the nightly's speed gain *was* the fp4 KV; both findings explained, nothing to promote.
- **acceptance stays ~0.29/0.81** → KV is not the cause; the acceptance change is elsewhere in the DSpark verify path and goes upstream as a bisect question.

## What runs (already written and staged — do not rewrite)

`scripts/t3b6-fp8kv-runner-2026-09-19.sh` on the box at `$BOX_HOME/pin-hot-experts/scripts/`. It is the T3b5 runner with one launch change: `KVDTYPE=fp8_ds_mla` → `--kv-cache-dtype fp8_ds_mla` (launcher `launch-t3-nightly.sh` gained a `KVDTYPE` passthrough; the Mac copy was re-synced from the box first). Same cache choreography (fast set copied to the canonical hash dir, slow set aside, everything restored at the end), same windows **T3b6 → v18ctl7 → T3b6b**, same instruments (knee ×2, knee6, `agent_fixture_o`, replay ×2, parity 18, power lines).

**Hash caveat.** The KV dtype may change the FlashInfer kernel-shape hash. If the boot log shows `Autotuning process starts` with a **long** tune and `Saved` into a **new** hash dir (not `ed692e15…`), record: new hash, tune minutes, and that the pinned cache did not apply. The run continues (bind budget 100 min). In that case T3b6 is read against v18ctl7 only, with T3b5 (188, nvfp4 KV, pinned) as the reference point for "what the cache pin was worth".

## Your job

1. Stage check on the box: only `dsv41-vllm-v18-cgsizes-BOUND-REF` running; the `103a/` dir shows `bba7410c…`, `ed692e15…`, `ed692e15….from-T3b4`, `ed692e15….after-T3b5`; no `STOP-CAMPAIGN`; `MemAvailable` > 100 GiB; `grep -n KVDTYPE $BOX_HOME/pin-hot-experts/scripts/launch-t3-nightly.sh` shows the passthrough.
2. Launch detached: `nohup bash $BOX_HOME/pin-hot-experts/scripts/t3b6-fp8kv-runner-2026-09-19.sh > $BOX_HOME/dsv41/t3b6-nohup.out 2>&1 &` + `disown`. Never foreground.
3. Poll every 2–3 min (`tail -5` campaign log; `ps -eo args | grep '^bash $BOX_HOME/pin-hot-experts/scripts/t3b6'`). Nothing else against `:30006`. Budget ~1.5 h if hash-hit, ~3 h if it retunes.
4. Read back — **fixture first**, from `agentfix-{T3b6,v18ctl7,T3b6b}.json` (`res.<class>.tok_s`, `.accept`, `.acc_per_step`, top-level `weighted_accept`): the verdict table is per-class acceptance and tok/s for prose / structured / code / shell_ops / tool_json, candidate vs control, plus T3b5's row copied from its README for the nvfp4 reference. Then knee (`knee-*-r{1,2}.json`, `knee6-*.json`; pair r1/r1, r2/r2), replay, boot facts (`boot-*.txt`: **the `KV cache format` line — it must say `fp8_ds_mla`**, `GPU KV cache size` — expect ~2.5M not 4.5M, `Loaded`/`Saved`, hash dir, rehome 206.61/62.33), parity (`e2c_compare_parity.py parity-A.json parity-B.json` prints `{exact, n}`; compare T3b6 vs T3b6b, T3b6 vs T3b5, T3b6 vs v18ctl7 — note whether fp8 KV moves the nightly closer to 0909's tokens), power lines.
5. Copy receipts to Mac `results/2026-09-19-overnight/T3b6/receipts/` (incl. `cachepin-sha-T3b6.txt` and any new-hash `autotune_configs.json` with sha256). Note: `run_window_T1T2.sh` leaves its files in `$BOX_HOME/dsv41/overnight-T1T2/`, not `overnight-T3/` — look there.
6. Write `results/2026-09-19-overnight/T3b6/README.md` (fixture table first, then knee, boot facts, parity, power) and ledger entries. End with a **VERDICT T3b6:** line naming which of the three outcomes it is.

## Bars (report; Milo promotes)

- Boot must log `fp8_ds_mla` — if it logs `nvfp4_ds_mla` the flag did not take: stop after the first window, report, restore v18.
- **KV-EXPLAINS-ACCEPTANCE:** prose accept ≥ 0.33 and tool_json accept ≥ 0.86 on both candidate windows (control ~0.35 / 0.88).
- **SPEED-WAS-THE-FP4-KV:** C1 within ±3% of v18ctl7 while acceptance recovered.
- **KV-NOT-THE-CAUSE:** acceptance within ±0.02 of T3b5's (0.288 / 0.81).
- Promotion-grade (report as such, do not promote): acceptance recovered **and** C1 ≥ +5% vs v18ctl7 **and** no fixture class < −3% **and** T3b6 vs T3b6b C1 within 2%.

## Stop / failure

As T3b5: boot fail → runner renames `-BOOT-FAIL`, restores caches and v18; record the exact error; do not retry. v18 not back after 15 min → `docker start dsv41-vllm-v18-cgsizes-BOUND-REF` once, report, stop. Do not touch `.from-T3b4`, `.after-T3b5`, `bba7410c…`, or any container not named here.

End with the VERDICT line and a ≤15-line summary: outcome, per-class acceptance cand vs ctl, C1/C4/C8/C16, KV format + size, hash hit or new hash, parity counts, power, what is up on `:30006`.
